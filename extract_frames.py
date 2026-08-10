import cv2
import random
import shutil
from pathlib import Path

VIDEO_DIR = Path("videos")
FRAME_DIR = Path("dataset/images")
ANNOTATION_DIR = Path("dataset/annotation_frames")

if FRAME_DIR.exists():
    shutil.rmtree(FRAME_DIR)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

if ANNOTATION_DIR.exists():
    shutil.rmtree(ANNOTATION_DIR)
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

frame_count = 0

for video_file in VIDEO_DIR.glob("*.mp4"):
    safe_name = video_file.name.encode('ascii', 'ignore').decode('ascii')
    print(f"Processing: {safe_name}")

    cap = cv2.VideoCapture(str(video_file))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print("Could not read FPS.")
        continue

    # Save one frame every 2 seconds
    interval = int(fps * 2)

    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % interval == 0:
            filename = FRAME_DIR / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            frame_count += 1

        count += 1

    cap.release()

print(f"\nDone! Extracted {frame_count} frames.")

# Select 200 frames randomly for annotation
all_frames = list(FRAME_DIR.glob("*.jpg"))
num_to_select = min(200, len(all_frames))

if num_to_select > 0:
    selected_frames = random.sample(all_frames, num_to_select)
    for frame_path in selected_frames:
        shutil.copy(frame_path, ANNOTATION_DIR / frame_path.name)
    print(f"Randomly selected {len(selected_frames)} frames and copied them to {ANNOTATION_DIR}")
else:
    print("No frames were extracted.")