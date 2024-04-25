import os
import subprocess

def extract_frames(video_dir, frame_rate_divider=24, quality_factor=95):
    # Create the output directory if it doesn't exist
    extracted_dir = os.path.join(video_dir, "extracted")
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)

    processes = []

    for filename in os.listdir(video_dir):
        if filename.endswith(".mkv"):  # or any other video format
            # Construct the base of the filename for output
            name_base = os.path.splitext(filename)[0]
            
            # Create a subfolder for the episode
            episode_dir = os.path.join(extracted_dir, name_base)
            if not os.path.exists(episode_dir):
                os.makedirs(episode_dir)

            # Construct the ffmpeg command
            cmd = (
                f'ffmpeg -i "{os.path.join(video_dir, filename)}" -vf '
                f'"select=not(mod(n\,{frame_rate_divider})),setpts=N/FRAME_RATE/TB" '
                f'-c:v libwebp -q:v {quality_factor} '
                f'-lossless 0 '
                f'"{os.path.join(episode_dir, f"{name_base}(%d).webp")}"'
            )
            
            # Start the command and move on
            process = subprocess.Popen(cmd, shell=True)
            processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.wait()

if __name__ == "__main__":
    extract_frames(r"D:\SD\Model Projects\Weakest Tamer")
