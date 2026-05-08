import os
import cv2
from tqdm import tqdm

def get_video_duration(video_path):
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    cap.release()
    return duration

def extract_frames_from_video(video_path, num_frames, start_frame_index, output_dir):
    """Extract specified number of frames from a single video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Video has 0 frames: {video_path}")
        return
    
    # Calculate sampling interval
    interval = total_frames / num_frames
    frame_indices = [int(i * interval) for i in range(num_frames)]
    
    current_frame = 0
    frame_count = 0
    
    for target_frame in tqdm(frame_indices, desc=f"Processing video {os.path.basename(video_path)}"):
        while current_frame <= target_frame:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            
        if ret:
            # Save frame using unified numbering system
            frame_filename = f"{start_frame_index + frame_count:06d}.png"
            output_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(output_path, frame)
            frame_count += 1
    
    cap.release()
    return frame_count

def distributed_frame_extraction(video_dir, output_dir, total_frames):
    """Perform distributed frame extraction from all videos in a directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all video files
    video_files = [f for f in sorted(os.listdir(video_dir)) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("No video files found")
        return
    
    # Calculate duration for each video
    video_durations = []
    total_duration = 0
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        duration = get_video_duration(video_path)
        video_durations.append(duration)
        total_duration += duration
    
    # Distribute frames based on duration ratio
    frames_per_video = []
    remaining_frames = total_frames
    for i, duration in enumerate(video_durations):
        if i == len(video_durations) - 1:
            # Last video gets all remaining frames
            frames = remaining_frames
        else:
            frames = int((duration / total_duration) * total_frames)
            remaining_frames -= frames
        frames_per_video.append(frames)
    
    print("Frame distribution among videos:")
    print(f"{'Video File':30} {'Duration (s)':15} {'Frames Assigned':15}")
    for video_file, duration, frames in zip(video_files, video_durations, frames_per_video):
        print(f"{video_file:30} {duration:<15.2f} {frames:<15}")
    
    # Execute frame extraction
    current_frame_index = 0
    for video_file, num_frames in zip(video_files, frames_per_video):
        video_path = os.path.join(video_dir, video_file)
        frames_extracted = extract_frames_from_video(
            video_path, num_frames, current_frame_index, output_dir)
        current_frame_index += frames_extracted
        
    print(f"Complete! Total frames extracted: {current_frame_index}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed video frame extraction tool")
    parser.add_argument("-i", "--input_dir", help="Path to video directory")
    parser.add_argument("-o", "--output_dir", help="Path to output directory")
    parser.add_argument("-t", "--total_frames", type=int, help="Total number of frames to extract")

    args = parser.parse_args()

    distributed_frame_extraction(args.input_dir, args.output_dir, args.total_frames)