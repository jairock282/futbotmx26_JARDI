import os
import shutil
import subprocess

def get_frames(
    video_file: str,
    raw_videos_dir_path: str,
    output_dir_path: str,
    fps: int = 10
):
    """
    Extract frames from a video file.
    """
    video_name = video_file.split("/")[-1].split(".")[0]
    video_path = os.path.join(raw_videos_dir_path, video_file)
    frames_output_dir_path = os.path.join(output_dir_path, video_name)
    os.makedirs(frames_output_dir_path, exist_ok=True)
    frame_output_path = os.path.join(frames_output_dir_path, "frame_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        frame_output_path,
    ]
    print(f"Extracting frames: {" ".join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("FFmpeg error:\n", result.stderr)
        if os.path.exists(frames_output_dir_path):
            shutil.rmtree(frames_output_dir_path)
        raise RuntimeError("FFmpeg failed")
    print(f"Frames extracted to: {frames_output_dir_path}\n")


def main():
    root_dir_path = "./data"
    raw_videos_dir = "raw_videos/"
    frames_output_dir = "frames/"

    raw_videos_dir_path = os.path.join(root_dir_path, raw_videos_dir)
    for video_file in os.listdir(raw_videos_dir_path):
        get_frames(
            video_file=video_file,
            raw_videos_dir_path=raw_videos_dir_path,
            output_dir_path=os.path.join(root_dir_path, frames_output_dir),
            fps=10
        )


if __name__ == "__main__":
    main()
