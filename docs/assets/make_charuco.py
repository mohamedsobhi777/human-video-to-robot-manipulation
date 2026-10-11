"""Printable A4 ChArUco board for recovering table pose and metric scale from phone video.

Board: 5 x 7 squares, 30 mm squares, 22 mm ArUco markers, DICT_4X4_50. Print at 100% scale
("actual size", no fit-to-page) and check the 100 mm bar with a ruler before recording.
"""
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SQUARES_X, SQUARES_Y, SQUARE_MM, MARKER_MM = 5, 7, 30.0, 22.0
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM/1000, MARKER_MM/1000,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
px_per_mm = 20
img = board.generateImage((int(SQUARES_X*SQUARE_MM*px_per_mm), int(SQUARES_Y*SQUARE_MM*px_per_mm)))

A4 = (210, 297)
fig = plt.figure(figsize=(A4[0]/25.4, A4[1]/25.4))
w, h = SQUARES_X*SQUARE_MM, SQUARES_Y*SQUARE_MM
left, bottom = (A4[0]-w)/2, A4[1]-20-h
ax = fig.add_axes([left/A4[0], bottom/A4[1], w/A4[0], h/A4[1]])
ax.imshow(img, cmap="gray", interpolation="nearest"); ax.axis("off")
bar = fig.add_axes([left/A4[0], 25/A4[1], 100/A4[0], 3/A4[1]])
bar.set_facecolor("black"); bar.set_xticks([]); bar.set_yticks([])
fig.text(left/A4[0], 32/A4[1], "100 mm: check with a ruler. Print at 100% / actual size.", fontsize=9)
fig.text(left/A4[0], 15/A4[1], f"ChArUco {SQUARES_X}x{SQUARES_Y}, square {SQUARE_MM:.0f} mm, marker {MARKER_MM:.0f} mm, DICT_4X4_50",
         fontsize=8)
out = Path(__file__).with_name("charuco_a4.pdf")
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=60)
print(out)
