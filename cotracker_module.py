import os
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path, Visualizer
import nvidia_smi  # Add this import
import cv2  # Add this import for cv2

class CoTrackerModule:
    def __init__(self, checkpoint_path: str):
        """
        Initialize CoTracker module
        
        Args:
            checkpoint_path (str): Path to CoTracker checkpoint file
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CoTracker model
        self.model = CoTrackerPredictor(checkpoint=checkpoint_path)
        self.model = self.model.to(self.device)
        
        # Initialize nvidia-smi
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    
    def print_gpu_usage(self):
        """Print current GPU memory usage"""
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        print(f"\nGPU Memory Usage:")
        print(f"Total: {info.total / 1024**3:.2f} GB")
        print(f"Used: {info.used / 1024**3:.2f} GB")
        print(f"Free: {info.free / 1024**3:.2f} GB")
        
    def process_video(self, 
                     video_path: str, 
                     grid_size: int,
                     start_frame: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process video using CoTracker to get motion traces
        
        Args:
            video_path (str): Path to video file
            grid_size (int): Size of grid (s in s^2 points)
            start_frame (int): Starting frame index t
            
        Returns:
            Tuple containing:
            - pred_tracks: Tensor of shape (B, T, N, 2) containing track coordinates
            - pred_visibility: Tensor of shape (B, T, N) containing visibility flags
            where:
                B = batch size (1)
                T = number of frames
                N = number of tracks (grid_size^2)
        """
        self.print_gpu_usage()  # Print memory usage before processing
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Read video in chunks to save memory
        video = read_video_from_path(video_path)
        chunk_size = 32  # Adjust this based on your GPU memory
        
        # Get video dimensions
        T, H, W, C = video.shape
        
        # Resize video if too large
        if H > 720 or W > 1280:  # Adjust these thresholds as needed
            scale = min(720/H, 1280/W)
            new_H, new_W = int(H * scale), int(W * scale)
            resized_video = []
            for frame in video:
                frame = cv2.resize(frame, (new_W, new_H))
                resized_video.append(frame)
            video = np.stack(resized_video)
        
        chunks = []
        
        for i in range(0, T, chunk_size):
            chunk = video[i:min(i+chunk_size, T)]
            chunk_tensor = torch.from_numpy(chunk).permute(0, 3, 1, 2)[None].float()
            
            if self.device == "cuda":
                chunk_tensor = chunk_tensor.cuda()
            
            # Process chunk
            with torch.cuda.amp.autocast():  # Use mixed precision
                pred_tracks_chunk, pred_visibility_chunk = self.model(
                    chunk_tensor, 
                    grid_size=grid_size
                )
            
            # Move results to CPU to free GPU memory
            chunks.append((
                pred_tracks_chunk.cpu(),
                pred_visibility_chunk.cpu()
            ))
            
            # Clear GPU memory after each chunk
            torch.cuda.empty_cache()
            self.print_gpu_usage()  # Monitor memory usage
        
        # Combine chunks
        pred_tracks = torch.cat([c[0] for c in chunks], dim=1)
        pred_visibility = torch.cat([c[1] for c in chunks], dim=1)
        
        # Create visualization in output/visualizations directory
        output_dir = os.path.dirname(os.path.dirname(os.path.dirname(video_path)))  # Go up to output dir
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        vis = Visualizer(save_dir=vis_dir)
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        if self.device == "cuda":
            video_tensor = video_tensor.cuda()
        
        # Generate visualization video with correct path
        clip_name = os.path.splitext(os.path.basename(video_path))[0]
        vis_path = clip_name + "_traces.mp4"  # Let Visualizer handle the full path
        
        vis.visualize(
            video=video_tensor,
            tracks=pred_tracks,
            visibility=pred_visibility,
            filename=vis_path
        )
        
        return pred_tracks, pred_visibility

def setup_cotracker(base_dir: str) -> Optional[str]:
    """
    Setup CoTracker by downloading checkpoint if needed
    
    Args:
        base_dir (str): Base directory for checkpoints
        
    Returns:
        Path to checkpoint file or None if setup fails
    """
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "scaled_offline.pth")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Download checkpoint if it doesn't exist
    if not os.path.exists(checkpoint_path):
        try:
            import wget
            url = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
            wget.download(url, checkpoint_path)
        except Exception as e:
            print(f"Failed to download checkpoint: {e}")
            return None
    
    return checkpoint_path 