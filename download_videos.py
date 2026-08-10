from pathlib import Path
import yt_dlp
import shutil

VIDEO_DIR = Path("videos")
if VIDEO_DIR.exists():
    shutil.rmtree(VIDEO_DIR)
VIDEO_DIR.mkdir(exist_ok=True)

with open("links.txt", "r") as f:
    urls = [line.strip() for line in f if line.strip()]

ydl_opts = {
    "format": "best[ext=mp4]/best",
    "outtmpl": str(VIDEO_DIR / "%(title)s.%(ext)s"),
    "merge_output_format": "mp4",
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for url in urls:
        try:
            print(f"\nDownloading: {url}")
            ydl.download([url])
        except Exception as e:
            print(f"Failed: {url}")
            print(e)

print("\nAll downloads completed.")