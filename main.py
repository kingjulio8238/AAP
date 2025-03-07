import argparse
from video_processor import VideoProcessor
from clip_filter import ClipFilter
import os
from pathlib import Path
from cotracker_module import CoTrackerModule, setup_cotracker
import torch

# Add these at the start of the script
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def process_video(video_path: str = "input.mp4", 
                 output_dir: str = "output",
                 text_annotations: list = ["bear"],
                 segment_duration: int = 30,
                 similarity_threshold: float = 0.25,
                 grid_size: int = 3):
    """
    Process video:
    1. Split into segments and scenes
    2. Track motion using CoTracker
    3. Filter clips based on text similarity
    
    Args:
        video_path (str): Path to input video
        output_dir (str): Output directory
        text_annotations (list): List of text annotations to match
        segment_duration (int): Duration of each initial segment
        similarity_threshold (float): Minimum similarity score for clips
        grid_size (int): Size of grid for motion tracking (s in s^2 points)
    """
    # Create all necessary directories at the start
    for subdir in ["checkpoints", "segments", "clips", "traces", "visualizations"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Verify input video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    
    # Setup CoTracker
    print("\nSetting up CoTracker...")
    checkpoint_path = setup_cotracker(output_dir)
    if checkpoint_path is None:
        raise RuntimeError("Failed to setup CoTracker")
    
    cotracker = CoTrackerModule(checkpoint_path)
    
    # Before heavy processing, clear GPU cache
    torch.cuda.empty_cache()
    
    # Set to use mixed precision
    torch.backends.cudnn.benchmark = True
    
    # Step 1: Process video into clips
    processor = VideoProcessor(video_path, output_dir, segment_duration)
    
    print("\nStep 1: Splitting video into segments...")
    segment_files = processor.split_into_segments()
    
    print("\nStep 2: Detecting and splitting scenes in segments...")
    clips_dirs = []
    for segment_file in segment_files:
        clips_dir = processor.detect_and_split_scenes(segment_file)
        clips_dirs.append(clips_dir)
        print(f"Created clips in: {clips_dir}")
    
    # Step 3: Process each clip with CoTracker
    print("\nStep 3: Extracting motion traces using CoTracker...")
    traces_dir = os.path.join(output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    
    for clips_dir in clips_dirs:
        for clip_path in Path(clips_dir).rglob("*.mp4"):
            pred_tracks, pred_visibility = cotracker.process_video(
                str(clip_path),
                grid_size=grid_size,
                start_frame=0
            )
            # Save traces for future use
            trace_path = os.path.join(traces_dir, f"{clip_path.stem}_traces.pt")
            torch.save({
                'tracks': pred_tracks,
                'visibility': pred_visibility
            }, trace_path)
    
    # Step 4: Filter clips using CLIP
    print("\nStep 4: Filtering clips based on text similarity...")
    clip_filter = ClipFilter(similarity_threshold=similarity_threshold)
    
    all_results = {}
    for clips_dir in clips_dirs:
        results = clip_filter.filter_clips(
            clips_dir, 
            text_annotations,
            num_frames=5,
            grid_size=3
        )
        
        # Merge results
        for text, clips in results.items():
            if text not in all_results:
                all_results[text] = []
            all_results[text].extend(clips)
    
    # Sort all results by similarity score
    for text in all_results:
        all_results[text].sort(key=lambda x: x[1], reverse=True)
    
    # Print final results
    print("\nResults:")
    print("-" * 50)
    for text, clips in all_results.items():
        print(f"\nText: {text}")
        for clip_path, score in clips:
            print(f"  {clip_path}: {score:.3f}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Process and filter video clips")
    parser.add_argument("--video", default="input.mp4",
                       help="Path to input video file (default: input.mp4)")
    parser.add_argument("--output", default="output",
                       help="Output directory (default: output)")
    parser.add_argument("--annotations", nargs="+", 
                       default=["bear"],
                       help="Text annotations to match (default: 'bear')")
    parser.add_argument("--segment-duration", type=int, default=30,
                       help="Duration of each initial segment in seconds (default: 30)")
    parser.add_argument("--similarity-threshold", type=float, default=0.25,
                       help="Minimum similarity score for clips (default: 0.25)")
    parser.add_argument("--grid-size", type=int, default=3,
                       help="Size of grid for motion tracking (default: 3)")
    
    args = parser.parse_args()
    
    try:
        results = process_video(
            args.video,
            args.output,
            args.annotations,
            args.segment_duration,
            args.similarity_threshold,
            args.grid_size
        )
        
        # Save results to file
        results_file = os.path.join(args.output, "results.txt")
        with open(results_file, "w") as f:
            for text, clips in results.items():
                f.write(f"\nText: {text}\n")
                for clip_path, score in clips:
                    f.write(f"  {clip_path}: {score:.3f}\n")
        
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 