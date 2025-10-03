import cv2
import yaml
import numpy as np

# === CONFIG ===
input_pgm = "map.pgm"          # default SLAM map
input_yaml = "map.yaml"
output_pgm = "new_map.pgm"     # final aligned map
output_yaml = "new_map.yaml"
resolution = 0.05              # meters per pixel
size_m = 6.0                   # arena size in meters
# ==============

# Load map
img = cv2.imread(input_pgm, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"❌ Could not load {input_pgm}")

(h, w) = img.shape
size_px = int(size_m / resolution)  # e.g. 120 px for 6 m

# --- Raw PGM writer ---
def save_pgm(filename, array):
    """Save numpy array as raw binary PGM (P5)."""
    h, w = array.shape
    maxval = 255
    with open(filename, "wb") as f:
        f.write(bytearray(f"P5\n{w} {h}\n{maxval}\n", "ascii"))
        f.write(array.tobytes())

# --- Trackbar callback ---
def nothing(x): pass

# Window + trackbars
cv2.namedWindow("Align Map (s=save, q=quit)")
cv2.createTrackbar("Angle", "Align Map (s=save, q=quit)", 180, 360, nothing)  # -180..180
cv2.createTrackbar("ShiftX", "Align Map (s=save, q=quit)", w//2, w, nothing)
cv2.createTrackbar("ShiftY", "Align Map (s=save, q=quit)", h//2, h, nothing)

while True:
    angle = cv2.getTrackbarPos("Angle", "Align Map (s=save, q=quit)") - 180
    shift_x = cv2.getTrackbarPos("ShiftX", "Align Map (s=save, q=quit)")
    shift_y = cv2.getTrackbarPos("ShiftY", "Align Map (s=save, q=quit)")

    # Rotate
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)

    # Define crop region
    x1 = shift_x - size_px//2
    y1 = shift_y - size_px//2
    x2 = shift_x + size_px//2
    y2 = shift_y + size_px//2

    # Pad if crop is outside boundaries
    crop = np.ones((size_px, size_px), dtype=np.uint8) * 205  # gray = unknown
    x1_src = max(0, x1); y1_src = max(0, y1)
    x2_src = min(w, x2); y2_src = min(h, y2)
    x1_dst = x1_src - x1; y1_dst = y1_src - y1
    x2_dst = x1_dst + (x2_src - x1_src)
    y2_dst = y1_dst + (y2_src - y1_src)

    crop[y1_dst:y2_dst, x1_dst:x2_dst] = rotated[y1_src:y2_src, x1_src:x2_src]

    # Preview with grid
    preview = cv2.resize(crop, (400,400), interpolation=cv2.INTER_NEAREST)
    for i in range(0, size_px, int(1/resolution)):  # 1m grid lines
        cv2.line(preview, (i*400//size_px, 0), (i*400//size_px, 400), (128,128,128), 1)
        cv2.line(preview, (0, i*400//size_px), (400, i*400//size_px), (128,128,128), 1)

    cv2.imshow("Align Map (s=save, q=quit)", preview)

    key = cv2.waitKey(50) & 0xFF
    if key == ord('s'):  # Save
        save_pgm(output_pgm, crop)

        # Update YAML
        with open(input_yaml, "r") as f:
            data = yaml.safe_load(f)
        data["image"] = output_pgm
        data["resolution"] = resolution
        data["origin"] = [0.0, 0.0, 0.0]

        with open(output_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"✅ Saved {output_pgm} (raw PGM) and {output_yaml} with rotation {angle}° and shift ({shift_x},{shift_y})")
        break
    elif key == ord('q'):
        print("❌ Quit without saving")
        break

cv2.destroyAllWindows()
