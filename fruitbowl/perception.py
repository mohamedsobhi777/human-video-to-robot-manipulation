"""Scene perception for the person's own recordings.

1. Camera calibration and table frame from the printed ChArUco board.
2. Zero-shot detection of the bowl and the three fruits (OWLv2), placed on
   the table by intersecting the viewing ray with a plane at the object's
   centre height.
3. Transcription of the spoken instruction (Whisper) and target-fruit lookup.

Table frame (metres): origin at the centre of the taped 30 x 40 cm rectangle,
+x away from the person, +y to their left, z up. The A4 board lies portrait,
printed side up, top of the page away from the person, centred on the
rectangle, with the bottom edge of the page on the rectangle's far tape line.
"""
import numpy as np
import cv2

# The board is designed with 30 mm squares and 22 mm markers. The printed copy came out
# at 90.6% (measured: board 136 x 190 mm, bar 90 mm), with 73 mm from the bottom edge of
# the page to the bottom edge of the squares.
PRINT_SCALE, MARGIN_M = 0.906, 0.073
SQUARES, SQUARE_M, MARKER_M = (5, 7), 0.030*PRINT_SCALE, 0.022*PRINT_SCALE
BOARD = cv2.aruco.CharucoBoard(SQUARES, SQUARE_M, MARKER_M, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
# Board frame: origin at the board's top-left corner, x to the person's right,
# y towards the person, z into the table. The rectangle's centre is 150 mm beyond
# the far tape line, where the bottom edge of the page lies.
RECT_CENTRE_B = np.array([SQUARES[0]*SQUARE_M/2, SQUARES[1]*SQUARE_M + MARGIN_M + 0.150, 0.0])
R_TABLE_FROM_BOARD = np.array([[0, -1.0, 0], [-1.0, 0, 0], [0, 0, -1.0]])


def detect_board(gray):
    det = cv2.aruco.CharucoDetector(BOARD)
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < 6: return None, None
    obj = BOARD.getChessboardCorners()[ids.ravel()]
    return obj.astype(np.float32), corners.reshape(-1, 2).astype(np.float32)


def calibrate(frames, focal_guess=None):
    """Intrinsics and board pose from frames of the static `calib` clip.

    With one fixed view, only the focal length is estimated (principal point
    at the image centre, square pixels, no distortion: modern phone video is
    already undistorted). Returns K, camera-from-table rotation and translation,
    and the reprojection error in pixels."""
    H, W = frames[0].shape[:2]
    objs, imgs = [], []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
        o, c = detect_board(gray)
        if o is not None: objs.append(o); imgs.append(c)
    if not objs: raise RuntimeError("calibration board not found")
    f0 = focal_guess or 0.9*max(W, H)
    K0 = np.array([[f0, 0, W/2], [0, f0, H/2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    err, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, (W, H), K0, np.zeros(5), flags=flags)
    R_cb = cv2.Rodrigues(rvecs[0])[0]; t_cb = tvecs[0].ravel()
    # camera-from-table: p_c = R_cb p_b + t_cb, with p_b = R_tb^T p_t + c_b
    R_ct = R_cb @ R_TABLE_FROM_BOARD.T
    t_ct = R_cb @ RECT_CENTRE_B + t_cb
    return dict(K=K, R_ct=R_ct, t_ct=t_ct, reproj_px=float(err), corners=int(np.mean([len(c) for c in imgs])))


def calibrate_views(frames):
    """Shared intrinsics from several fixed-board views ({name: rgb}) and the board pose in
    each: {name: calib dict} in the format of `calibrate`."""
    names = [n for n in frames]
    H, W = frames[names[0]].shape[:2]
    objs, imgs, used = [], [], []
    for n in names:
        o, c = detect_board(cv2.cvtColor(frames[n], cv2.COLOR_RGB2GRAY))
        if o is not None: objs.append(o); imgs.append(c); used.append(n)
    if not objs: raise RuntimeError("calibration board not found")
    f0 = 0.9*max(W, H)
    K0 = np.array([[f0, 0, W/2], [0, f0, H/2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    err, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, (W, H), K0, np.zeros(5), flags=flags)
    out = {}
    for n, rv, tv, c in zip(used, rvecs, tvecs, imgs):
        R_cb = cv2.Rodrigues(rv)[0]; t_cb = tv.ravel()
        out[n] = dict(K=K, R_ct=R_cb @ R_TABLE_FROM_BOARD.T, t_ct=R_cb @ RECT_CENTRE_B + t_cb, reproj_px=float(err), corners=len(c))
    return out


def camera_to_table(p_c, calib):
    """Camera-frame points (..., 3) -> table frame."""
    return (p_c - calib["t_ct"]) @ calib["R_ct"]


def pixel_to_table(uv, calib, height):
    """Intersect the viewing ray through pixel(s) uv (..., 2) with the plane z = height."""
    K, R, t = calib["K"], calib["R_ct"], calib["t_ct"]
    uv = np.atleast_2d(uv)
    rays_c = np.concatenate([(uv - K[:2, 2])/np.diag(K)[:2], np.ones((len(uv), 1))], 1)
    origin = camera_to_table(np.zeros(3), calib)
    dirs = rays_c @ R                                    # camera -> table rotation (R^T applied to rows)
    s = (height - origin[2])/dirs[:, 2]
    return origin + s[:, None]*dirs


# ---------------------------------------------------------------------------
QUERIES = {"persimmon": "a persimmon", "pear": "a pear", "mandarin": "a mandarin orange", "bowl": "a bowl"}
CENTRE_HEIGHT = {"persimmon": 0.03, "pear": 0.04, "mandarin": 0.025, "bowl": 0.03}


class Detector:
    """OWLv2 zero-shot detection with a scene prior: exactly one bowl and three
    fruits are on the table. Candidate fruit boxes are the three most fruit-like
    non-overlapping boxes; names are then assigned to them by an optimal
    one-to-one matching over OWLv2's per-name scores, so two names never share
    a fruit and a name never goes unassigned."""

    def __init__(self, device="cuda", model="google/owlv2-base-patch16-ensemble"):
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.torch, self.device = torch, device
        self.proc = Owlv2Processor.from_pretrained(model)
        self.model = Owlv2ForObjectDetection.from_pretrained(model).to(device).eval()

    def __call__(self, rgb, fruits=("persimmon", "pear", "mandarin"), prototypes=None):
        """prototypes: optional {name: embedding} from the calib clip; when given,
        names are matched by appearance similarity instead of text scores."""
        from scipy.optimize import linear_sum_assignment
        torch = self.torch
        texts = [QUERIES[n] for n in fruits] + ["a fruit", QUERIES["bowl"]]
        inp = self.proc(text=[texts], images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
        logits = out.logits[0].sigmoid().cpu().numpy()                       # (boxes, queries)
        H, W = rgb.shape[:2]
        size = max(H, W)                                                    # OWLv2 pads to a square
        cx, cy, w, h = out.pred_boxes[0].cpu().numpy().T*size
        boxes = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], 1)
        bowl = int(np.argmax(logits[:, -1]))
        result = {"bowl": dict(box=boxes[bowl], score=float(logits[bowl, -1]))}
        fruitness = logits[:, :-1].max(1)
        picked = []
        for i in np.argsort(-fruitness):
            if iou(boxes[i], boxes[bowl]) > 0.3 or any(iou(boxes[i], boxes[j]) > 0.3 for j in picked): continue
            picked.append(i)
            if len(picked) == len(fruits): break
        emb = out.class_embeds[0].cpu().numpy()
        emb = emb/np.linalg.norm(emb, axis=1, keepdims=True)
        if prototypes is not None:
            score = np.stack([emb[picked] @ prototypes[n] for n in fruits], 1)
        else:
            score = logits[np.ix_(picked, range(len(fruits)))]
        rows, cols = linear_sum_assignment(-score)
        for r, c in zip(rows, cols):
            result[fruits[c]] = dict(box=boxes[picked[r]], score=float(score[r, c]), embedding=emb[picked[r]])
        return result

    def prototypes_from_calib(self, rgb, order=("persimmon", "pear", "mandarin"), calib=None):
        """Appearance prototypes from the calib clip, where the fruits stand in a row
        in front of the bowl in a fixed order, left to right as seen by the person
        (decreasing table y). Uses the three fruit boxes found without names."""
        det = self(rgb)
        boxes = [det[n] for n in ("persimmon", "pear", "mandarin")]
        if calib is not None:
            ys = [pixel_to_table(np.array([(b["box"][0]+b["box"][2])/2, (b["box"][1]+b["box"][3])/2]), calib, 0.03)[0, 1] for b in boxes]
            ranked = [boxes[i] for i in np.argsort(-np.array(ys))]
        else:
            ranked = sorted(boxes, key=lambda b: b["box"][0])
        return {n: b["embedding"] for n, b in zip(order, ranked)}


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0)*max(0, y1 - y0)
    return inter/((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


def locate(detections, calib):
    """Table-frame xy of each detected object (box centre ray at its centre height)."""
    out = {}
    for n, d in detections.items():
        x0, y0, x1, y1 = d["box"]
        out[n] = pixel_to_table(np.array([(x0 + x1)/2, (y0 + y1)/2]), calib, CENTRE_HEIGHT[n])[0, :2]
    return out


def transcribe(video_path, model="small"):
    """Whisper transcript of the clip's audio, primed with the task vocabulary."""
    import whisper
    return whisper.load_model(model).transcribe(str(video_path), language="en",
                                                initial_prompt="Persimmon, pear, mandarin, bowl.")["text"].strip()


def target_from_text(text):
    t = text.lower()
    # Includes common mishearings of the short word "pear" (seen: "pier").
    words = dict(persimmon=["persimmon", "kaki"], pear=["pear", "pier", "pair", "peer", "nashi"],
                 mandarin=["mandarin", "orange", "tangerine", "clementine"])
    hits = {}
    for n, ws in words.items():
        found = [t.find(w) for w in ws if t.find(w) >= 0]
        if found: hits[n] = min(found)
    return min(hits, key=hits.get) if hits else None
