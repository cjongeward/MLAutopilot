import cv2
import os

def extract_frames(video_path, timestamps, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}")
    print(f"Total frames in video: {total_frames}")

    for timestamp in timestamps:
        frame_index = int((timestamp / 1000) * fps)
        if frame_index >= total_frames:
            print(f"Timestamp {timestamp} ms is beyond the video duration.")
            continue

        # Set the current frame position of the video file
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:
            # Save the frame as an image file
            frame_filename = os.path.join(output_folder, f"{timestamp}ms.png")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
        else:
            print(f"Failed to extract frame at {timestamp} ms")

    cap.release()


#cwd = os.getcwd()
#video_path = os.path.join(cwd, "..", "data", "video.mkv")
#output_folder = os.path.join(cwd, "..", "data", "temp")
#timestamps = [1000, 2500, 3000]
#extract_frames(video_path, timestamps, output_folder)
