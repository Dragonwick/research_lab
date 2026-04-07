"""
Logitech slot mapper — click each slot center on the live feed.
Order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
Press ENTER to save, ESC to redo from slot 1.
"""

import subprocess, json, numpy as np, cv2

SLOT_COUNT  = 11
VIDEO_DEV   = "/dev/video4"
OUTPUT_FILE = "/home/aaron/Documents/ot2files/logitech_slot_positions.json"
WIDTH, HEIGHT = 1280, 720

clicks = {}
current = [1]  # mutable so mouse callback can modify it


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and current[0] <= SLOT_COUNT:
        slot = current[0]
        clicks[slot] = (x, y)
        print(f"  Slot {slot} → ({x}, {y})")
        current[0] += 1
        if current[0] > SLOT_COUNT:
            print("All 11 done — press ENTER to save, ESC to redo.")


cmd = ["ffmpeg", "-f", "v4l2", "-video_size", f"{WIDTH}x{HEIGHT}",
       "-input_format", "mjpeg", "-i", VIDEO_DEV,
       "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"]
pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        bufsize=10**8)

cv2.namedWindow("Logitech Slot Mapper")
cv2.setMouseCallback("Logitech Slot Mapper", on_mouse)

print("Click each slot center in order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")
print("ENTER = save   ESC = redo from slot 1\n")

while True:
    raw = pipe.stdout.read(WIDTH * HEIGHT * 3)
    if len(raw) != WIDTH * HEIGHT * 3:
        break
    frame = np.frombuffer(raw, dtype="uint8").reshape((HEIGHT, WIDTH, 3)).copy()

    # Draw clicked slots
    for slot, (cx, cy) in clicks.items():
        cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)
        cv2.putText(frame, str(slot), (cx - 7, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Show next slot to click
    if current[0] <= SLOT_COUNT:
        msg = f"Click Slot {current[0]}  ({current[0]-1}/{SLOT_COUNT} done)"
    else:
        msg = "All done — ENTER to save, ESC to redo"
    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Logitech Slot Mapper", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 13 and current[0] > SLOT_COUNT:  # ENTER
        with open(OUTPUT_FILE, "w") as f:
            json.dump({str(k): list(v) for k, v in clicks.items()}, f, indent=2)
        print(f"\nSaved {len(clicks)} slot positions to {OUTPUT_FILE}")
        break
    elif key == 27:  # ESC
        clicks.clear()
        current[0] = 1
        print("Reset — start from slot 1 again.")

pipe.terminate()
cv2.destroyAllWindows()
