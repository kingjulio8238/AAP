#!/usr/bin/env python3
"""
Script to split a video into segments and then use PySceneDetect to further split each segment
into shorter clips with consistent shots.

Requirements:
- FFmpeg (https://ffmpeg.org/documentation.html)
- PySceneDetect (https://pyscenedetect.readthedocs.io/en/latest/)

Usage:
    python split_and_detect.py input_video.mp4 --segment_duration 300

This will split 'input_video.mp4' into segments of 300 seconds (5 minutes) each,
then run PySceneDetect on each segment using the 'detect-content' algorithm and split the video accordingly.
"""

import os
import glob
import argparse
import subprocess
from moviepy.editor import VideoFileClip
from scenedetect import detect, ContentDetector, split_video_ffmpeg
import cv2
import numpy as np

def split_video(input_file, segment_duration, output_dir):
    """
    Splits the input video into fixed-length segments using FFmpeg.
    
    Parameters:
        input_file (str): Path to the input video.
        segment_duration (int): Duration (in seconds) of each segment.
        output_dir (str): Directory where segments will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    segment_pattern = os.path.join(output_dir, "segment_%03d.mp4")
    command = [
        "ffmpeg",
        "-i", input_file,
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(segment_duration),
        "-f", "segment",
        segment_pattern
    ]
    print("Running FFmpeg command to split video:")
    print(" ".join(command))
    subprocess.run(command, check=True)

def process_segment(segment_file, output_dir):
    """
    Uses PySceneDetect to detect scene boundaries and split the segment video accordingly.
    
    Parameters:
        segment_file (str): Path to the segment video file.
        output_dir (str): Directory where the scene clips will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    # The command below uses the 'detect-content' detector and then splits the video by scenes.
    command = [
        "scenedetect",
        "-i", segment_file,
        "-o", output_dir,
        "detect-content",
        "split-video"
    ]
    print(f"Running PySceneDetect on {segment_file}:")
    print(" ".join(command))
    subprocess.run(command, check=True)

class VideoProcessor:
    def __init__(self, video_path, output_dir, segment_duration=1):
        """
        Initialize VideoProcessor
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save processed clips
            segment_duration (int): Duration of each initial segment in seconds (default: 1)
        """
        self.video_path = video_path
        self.base_output_dir = output_dir
        self.segment_duration = segment_duration
        
        # Create only necessary output directories
        self.segments_dir = os.path.join(output_dir, "segments")
        self.clips_dir = os.path.join(output_dir, "clips")
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)

    def split_into_segments(self):
        """Split video into 1-second segments"""
        video = VideoFileClip(self.video_path)
        duration = video.duration
        segment_files = []

        for start_time in range(0, int(duration)):  # Step by 1 second
            end_time = min(start_time + 1, duration)  # 1-second segments
            
            # Extract segment
            segment = video.subclip(start_time, end_time)
            
            # Generate output filename
            output_filename = f"segment_{int(start_time):04d}_{int(end_time):04d}.mp4"
            output_path = os.path.join(self.segments_dir, output_filename)
            
            # Write segment to file with higher quality settings
            segment.write_videofile(output_path, 
                                 codec='libx264',
                                 audio_codec='aac',
                                 temp_audiofile='temp-audio.m4a',
                                 remove_temp=True,
                                 fps=30)  # Ensure consistent framerate
            
            segment_files.append(output_path)
            print(f"Created segment: {output_filename}")

        video.close()
        return segment_files

    def detect_and_split_scenes(self, segment_path):
        """
        Detect and split scenes in a video segment using more sensitive detection
        """
        segment_name = os.path.splitext(os.path.basename(segment_path))[0]
        segment_clips_dir = os.path.join(self.clips_dir, segment_name)
        os.makedirs(segment_clips_dir, exist_ok=True)

        # Use more sensitive scene detection
        scene_list = detect(segment_path, ContentDetector(threshold=20.0))  # Lower threshold for more sensitive detection
        
        if not scene_list:
            # If no scenes detected, use the whole segment
            video = VideoFileClip(segment_path)
            scene_list = [(0, video.duration)]
            video.close()
        
        # Split video into scenes
        clips = []
        for i, (start_time, end_time) in enumerate(scene_list):
            output_path = os.path.join(segment_clips_dir, f"{segment_name}_scene_{i+1:03d}.mp4")
            
            # Extract scene with higher quality
            video_clip = VideoFileClip(segment_path)
            scene_clip = video_clip.subclip(start_time, end_time)
            scene_clip.write_videofile(output_path,
                                     codec='libx264',
                                     audio_codec='aac',
                                     temp_audiofile='temp-audio.m4a',
                                     remove_temp=True,
                                     fps=30)  # Ensure consistent framerate
            
            video_clip.close()
            clips.append(output_path)
            
        print(f"Split {segment_name} into {len(scene_list)} scenes")
        return segment_clips_dir

def main():
    parser = argparse.ArgumentParser(description="Process video into segments and scene-based clips")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Output directory for processed videos")
    parser.add_argument("--segment-duration", type=int, default=30,
                       help="Duration of each initial segment in seconds (default: 30)")
    
    args = parser.parse_args()

    # Initialize processor
    processor = VideoProcessor(args.video, args.output, args.segment_duration)
    
    # Step 1: Split video into segments
    print("\nStep 1: Splitting video into segments...")
    segment_files = processor.split_into_segments()
    
    # Step 2: Detect and split scenes in each segment
    print("\nStep 2: Detecting and splitting scenes in segments...")
    for segment_file in segment_files:
        clips_dir = processor.detect_and_split_scenes(segment_file)
        print(f"Created clips in: {clips_dir}")

if __name__ == "__main__":
    main()
