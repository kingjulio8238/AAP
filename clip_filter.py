import torch
from PIL import Image
import clip
import os
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict

class ClipFilter:
    def __init__(self, similarity_threshold=0.25):
        """
        Initialize CLIP model for filtering video clips based on text similarity
        
        Args:
            similarity_threshold (float): Minimum similarity score to keep clip-text pair
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.similarity_threshold = similarity_threshold

    def extract_middle_frame(self, video_path: str) -> Image.Image:
        """Extract middle frame from video clip"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)
        return None

    def extract_frames_with_grid(self, video_path: str, start_time_step: int = 0, num_frames: int = 5, grid_size: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Given video clip with l frames {I1, I2, ..., Il} ∈ R(l)×H×W×3,
        start from time step t and put a grid of equally spaced s^2 points on It
        
        Args:
            video_path (str): Path to video clip
            start_time_step (int): Starting time step t
            num_frames (int): Number of frames to sample after t
            grid_size (int): Size of grid (s in s^2 points)
        
        Returns:
            List of tuples (frame, grid_points) where:
            - frame is the image array of shape (H,W,3)
            - grid_points is array of shape (s^2, 2) containing (x,y) coordinates
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure start_time_step is valid
        start_time_step = min(max(0, start_time_step), total_frames - 1)
        
        # Calculate frame indices starting from t
        frame_indices = np.linspace(
            start_time_step, 
            min(total_frames-1, start_time_step + num_frames), 
            num_frames, 
            dtype=int
        )
        
        frames_and_grids = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Get frame dimensions
                height, width = frame.shape[:2]
                
                # Create grid points
                x_points = np.linspace(0, width-1, grid_size, dtype=int)
                y_points = np.linspace(0, height-1, grid_size, dtype=int)
                
                # Create all grid point combinations
                xx, yy = np.meshgrid(x_points, y_points)
                grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)  # shape: (s^2, 2)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Store both frame and grid points
                frames_and_grids.append((
                    frame_rgb,
                    grid_points
                ))
        
        cap.release()
        return frames_and_grids

    def compute_similarity(self, frames_and_grids: List[Tuple[np.ndarray, np.ndarray]], text: str) -> float:
        """
        Compute similarity using frames and their grid points
        
        Args:
            frames_and_grids: List of (frame, grid_points) tuples
            text (str): Text to compare against
        """
        with torch.no_grad():
            # Convert frames to PIL Images and preprocess
            images = [Image.fromarray(frame) for frame, _ in frames_and_grids]
            image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            text_input = clip.tokenize([text]).to(self.device)

            # Get features
            image_features = self.model.encode_image(image_inputs)
            text_features = self.model.encode_text(text_input)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity for each frame and average
            similarities = (image_features @ text_features.T).squeeze()
            avg_similarity = similarities.mean().item()

        return avg_similarity

    def filter_clips(self, clips_dir: str, text_annotations: List[str], 
                    start_time_step: int = 0, num_frames: int = 5, 
                    grid_size: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """
        Filter video clips based on similarity with text annotations
        """
        results = {text: [] for text in text_annotations}
        
        for clip_file in Path(clips_dir).rglob("*.mp4"):
            frames_and_grids = self.extract_frames_with_grid(
                str(clip_file), 
                start_time_step=start_time_step,
                num_frames=num_frames, 
                grid_size=grid_size
            )
            
            if not frames_and_grids:
                continue
                
            for text in text_annotations:
                similarity = self.compute_similarity(frames_and_grids, text)
                
                if similarity >= self.similarity_threshold:
                    results[text].append((str(clip_file), similarity))
        
        # Sort clips by similarity score
        for text in results:
            results[text].sort(key=lambda x: x[1], reverse=True)
            
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter video clips using CLIP-based similarity")
    parser.add_argument("--clips-dir", required=True, help="Directory containing video clips")
    parser.add_argument("--annotations", nargs="+", required=True, help="Text annotations to match")
    parser.add_argument("--threshold", type=float, default=0.25, help="Similarity threshold")
    
    args = parser.parse_args()
    
    clip_filter = ClipFilter(similarity_threshold=args.threshold)
    results = clip_filter.filter_clips(args.clips_dir, args.annotations)
    
    # Print results
    for text, clips in results.items():
        print(f"\nText: {text}")
        for clip_path, score in clips:
            print(f"  {clip_path}: {score:.3f}")

if __name__ == "__main__":
    main() 