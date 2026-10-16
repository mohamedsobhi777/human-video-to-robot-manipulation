"""From the person's phone clips to object-relative plans (same format as
egodex.human_plans), using the calibrated table frame, which is also the
simulator's frame (origin at the work rectangle's centre).

Per session: `calib` gives the camera intrinsics and table pose (ChArUco)
and appearance prototypes for the three fruits (fixed order in that clip).
Per clip: spoken instruction -> target fruit; first still frames -> layout of
bowl and fruits; WiLoR hand keypoints -> pinch path, aperture, yaw in the
table frame; aperture -> grasp and release.
"""
import json, subprocess
from pathlib import Path
import numpy as np
from fruitbowl import perception, handtrack, retarget


def video_info(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames:stream_side_data=rotation", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return dict(width=s["width"], height=s["height"], fps=num/den)


# iPhone HDR video (HLG, BT.2020) tone-mapped to standard colour, so detection sees normal images.
TONEMAP = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv"


def is_hdr(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_transfer",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    return out in ("arib-std-b67", "smpte2084")


def read_frames(path, fps=30, width=None, start=0.0, count=None):
    """Decode at a fixed frame rate (ffmpeg applies the phone's rotation flag), at native
    size unless `width` is given, tone-mapping HDR clips."""
    vf = ([TONEMAP] if is_hdr(path) else []) + [f"fps={fps}"] + ([f"scale={width}:-2"] if width else [])
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path), "-vf", ",".join(vf)]
    # Frame size after rotation and scaling, from one decoded PNG.
    import imageio.v3 as iio
    png = subprocess.run(cmd + ["-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    H, width = iio.imread(png).shape[:2]
    raw = subprocess.run(cmd + (["-frames:v", str(count)] if count else []) + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, width, 3), fps


def calibrate_clips(paths, seconds=0.3):
    """One focal length for the whole session (same phone and lens) and a board pose per
    clip, from a still frame at the start of each clip. Lets the phone move a little
    between clips as long as the board does not."""
    first = {Path(p).stem: read_frames(p, start=seconds, count=1)[0][0] for p in paths}
    return perception.calibrate_views(first), first


def calibrate_session(calib_path, detector):
    frames, fps = read_frames(calib_path)
    calib = perception.calibrate(list(frames[:int(3*fps):int(fps//2)]))
    protos = detector.prototypes_from_calib(frames[-int(fps)], calib=calib)
    return calib, protos


def plan_from_clip(path, calib, protos, detector, tracker, target=None, transcribe=True, frames=None):
    frames, fps = frames if frames is not None else read_frames(path)
    instruction = perception.transcribe(path) if transcribe else ""
    spoken = perception.target_from_text(instruction)
    target = spoken or target
    # Layout from a still frame before the hand moves (the clip starts with 2 s of stillness).
    still = frames[int(0.5*fps)]
    det = detector(still, prototypes=protos)
    loc = perception.locate(det, calib)
    # Hand: camera-frame keypoints -> table frame.
    kp = handtrack.track(frames, calib["K"], tracker, side="right")
    kp_t = perception.camera_to_table(kp, calib)
    thumb, index = kp_t[:, handtrack.THUMB_TIP], kp_t[:, handtrack.INDEX_TIP]
    both, info = retarget.fill_and_smooth(np.concatenate([thumb, index], 1))
    thumb, index = both[:, :3], both[:, 3:]
    pinch, aperture = (thumb + index)/2, np.linalg.norm(thumb - index, axis=1)
    g, r = retarget.find_events(pinch, aperture, fps=fps, use_aperture=True)
    fruits = ("persimmon", "pear", "mandarin")
    return dict(clip=Path(path).stem, instruction=instruction, target=fruits.index(target), target_name=target,
                spoken_target=spoken, pinch=pinch, aperture=aperture, yaw=retarget.closing_yaw(thumb, index), g=g, r=r,
                bowl_xy=loc["bowl"], fruits_xy=np.stack([loc[n] for n in fruits]), found=info["found"],
                detection_scores={n: d["score"] for n, d in det.items()}, fps=fps)
