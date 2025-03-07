import os
import torch
import numpy as np
import cv2
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker_module import setup_cotracker, CoTrackerModule

def process_video_with_cotracker(
    video_path: str,
    output_dir: str = "output",
    grid_size: int = 15,
    max_height: int = 270,
    mask_path: str = None  # Add mask path parameter
):
    """
    Apply CoTracker to a video file and save visualization - Optimized for A100
    
    Args:
        video_path (str): Path to input video
        output_dir (str): Output directory for results
        grid_size (int): Size of tracking grid (default: 15)
        max_height (int): Maximum height for video processing (default: 270)
        mask_path (str): Path to segmentation mask (optional)
    """
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    # Initialize CoTracker
    print("\nSetting up CoTracker...")
    checkpoint_path = setup_cotracker(output_dir)
    if checkpoint_path is None:
        raise RuntimeError("Failed to setup CoTracker")
    
    # Force garbage collection before starting
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Set memory allocation strategy to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Configure PyTorch for A100
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize CoTracker after memory cleanup
    cotracker = CoTrackerModule(checkpoint_path)
    
    print("\nProcessing video...")
    # Read video
    video = read_video_from_path(video_path)
    
    # Get video dimensions
    T, H, W, C = video.shape
    print(f"Original video shape: {video.shape}")
    
    # Resize if needed
    if H > max_height:
        scale = max_height / H
        new_H = max_height
        new_W = int(W * scale)
        print(f"Resizing video from {H}x{W} to {new_H}x{new_W}")
        resized_video = []
        for frame in video:
            frame = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_AREA)
            resized_video.append(frame)
        video = np.stack(resized_video)
        print(f"New video shape: {video.shape}")
    
    # Load segmentation mask if provided
    segm_mask = None
    if mask_path and os.path.exists(mask_path):
        print(f"Loading segmentation mask from {mask_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize mask to match video dimensions if needed
        if mask.shape[:2] != (video.shape[1], video.shape[2]):
            mask = cv2.resize(mask, (video.shape[2], video.shape[1]), interpolation=cv2.INTER_NEAREST)
        
        # Convert to torch tensor
        segm_mask = torch.from_numpy(mask).float() / 255.0
        
        # Expand dimensions to match CoTracker input format
        segm_mask = segm_mask.unsqueeze(0)  # Add batch dimension
        
        if torch.cuda.is_available():
            segm_mask = segm_mask.cuda()
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"\nUsing GPU: {torch.cuda.get_device_name()}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    cotracker.print_gpu_usage()
    
    # Convert video to tensor with lower precision to save memory
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].half()  # Use half precision
    if torch.cuda.is_available():
        video_tensor = video_tensor.cuda()
    
    print("\nRunning CoTracker...")
    # Process with mixed precision
    with torch.amp.autocast('cuda', dtype=torch.float16):
        pred_tracks, pred_visibility = cotracker.model(
            video_tensor, 
            grid_size=grid_size,
            segm_mask=segm_mask  # Pass the segmentation mask
        )
    
    # Move results to CPU to free GPU memory
    pred_tracks = pred_tracks.cpu()
    pred_visibility = pred_visibility.cpu()
    
    # Free up GPU memory
    del video_tensor
    if segm_mask is not None:
        del segm_mask
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create visualization
    print("\nCreating visualization...")
    vis = Visualizer(
        save_dir=os.path.join(output_dir, "visualizations"),
        fps=30
    )
    output_name = os.path.splitext(os.path.basename(video_path))[0] + "_tracked.mp4"
    
    # Convert video back to tensor for visualization
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    
    vis.visualize(
        video=video_tensor,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename=output_name
    )
    
    # Save tracking data
    trace_path = os.path.join(output_dir, "traces", f"{os.path.splitext(os.path.basename(video_path))[0]}_traces.pt")
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    torch.save({
        'tracks': pred_tracks,
        'visibility': pred_visibility,
        'video_shape': video.shape,
        'grid_size': grid_size
    }, trace_path)
    
    print(f"\nVisualization saved to: {os.path.join(output_dir, 'visualizations', output_name)}")
    print(f"Tracking data saved to: {trace_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Apply CoTracker to video (A100 Optimized)")
    parser.add_argument("--video", default="input.mp4",
                       help="Path to input video file (default: input.mp4)")
    parser.add_argument("--output", default="output",
                       help="Output directory (default: output)")
    parser.add_argument("--grid-size", type=int, default=15,
                       help="Size of tracking grid (default: 15)")
    parser.add_argument("--max-height", type=int, default=270,
                       help="Maximum height for video processing (default: 270)")
    parser.add_argument("--mask", default=None,
                       help="Path to segmentation mask (optional)")
    
    args = parser.parse_args()
    
    try:
        process_video_with_cotracker(
            args.video,
            args.output,
            args.grid_size,
            args.max_height,
            args.mask
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 