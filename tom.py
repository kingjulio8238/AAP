import torch
import numpy as np
from siloed_cotracker import process_video_with_cotracker
from homography_utils import remove_global_motion
from sklearn.cluster import KMeans
import os
import cv2
from PIL import Image

# Import SoM-related modules
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import inference_seem_pano

# For semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto

class ToMGenerator:
    def __init__(self, grid_size=15, global_motion_threshold=2.0, foreground_threshold=2.0):
        """
        Initialize ToM Generator.
        
        Args:
            grid_size: Size of tracking grid (s=15)
            global_motion_threshold: Threshold for global motion detection (η=2)
            foreground_threshold: Threshold for foreground classification (ϵ=2)
        """
        self.grid_size = grid_size
        self.global_motion_threshold = global_motion_threshold
        self.foreground_threshold = foreground_threshold
        
        # Initialize SoM models
        self.init_som_models()
        
    def init_som_models(self):
        """Initialize the SoM models for applying marks to frames"""
        # Load configurations
        semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
        seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

        semsam_ckpt = "./swinl_only_sam_many2many.pth"
        sam_ckpt = "./sam_vit_h_4b8939.pth"
        seem_ckpt = "./seem_focall_v1.pt"

        # Load options
        self.opt_semsam = load_opt_from_config_file(semsam_cfg)
        self.opt_seem = load_opt_from_config_file(seem_cfg)
        self.opt_seem = init_distributed_seem(self.opt_seem)

        # Build models
        self.model_semsam = BaseModel(self.opt_semsam, build_model(self.opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
        self.model_seem = BaseModel_Seem(self.opt_seem, build_model_seem(self.opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

        # Initialize text embeddings for SEEM model
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                self.model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                    COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
    
    def has_global_motion(self, tracks, visibility=None):
        """
        Check if tracks contain global motion.
        
        Args:
            tracks: Tensor of shape (B, T, N, 2)
            visibility: Optional tensor of shape (B, T, N)
        Returns:
            Boolean indicating presence of global motion
        """
        # Compute average motion between consecutive frames
        motion = torch.diff(tracks, dim=1)  # (B, T-1, N, 2)
        if visibility is not None:
            valid_vis = visibility[:, 1:] * visibility[:, :-1]  # (B, T-1, N)
            motion = motion * valid_vis.unsqueeze(-1)
        
        # Compute average motion magnitude
        motion_mag = torch.norm(motion, dim=-1)  # (B, T-1, N)
        avg_motion = motion_mag.mean()
        
        return avg_motion > self.global_motion_threshold
    
    def classify_traces(self, tracks, visibility=None):
        """
        Classify traces into foreground and background based on motion magnitude.
        
        Args:
            tracks: Tensor of shape (B, T, N, 2)
            visibility: Optional tensor of shape (B, T, N)
        Returns:
            foreground_tracks, background_tracks
        """
        B, T, N, _ = tracks.shape
        
        # Compute motion magnitude for each track
        motion = torch.diff(tracks, dim=1)  # (B, T-1, N, 2)
        motion_mag = torch.norm(motion, dim=-1)  # (B, T-1, N)
        
        # Average motion magnitude across time
        avg_motion = motion_mag.mean(dim=1)  # (B, N)
        
        # Apply visibility weights if provided
        if visibility is not None:
            # Only consider points that are visible in consecutive frames
            vis_weight = visibility[:, 1:] * visibility[:, :-1]  # (B, T-1, N)
            # Average visibility across time
            vis_weight = vis_weight.float().mean(dim=1)  # (B, N)
            # Apply visibility weight to motion magnitude
            avg_motion = avg_motion * vis_weight
        
        # Classify based on motion magnitude threshold (ϵ)
        fg_mask = avg_motion > self.foreground_threshold
        
        # Split tracks into foreground and background
        foreground_tracks = []
        background_tracks = []
        
        for b in range(B):
            fg_indices = torch.where(fg_mask[b])[0]
            bg_indices = torch.where(~fg_mask[b])[0]
            
            # Extract foreground and background tracks
            fg_track = tracks[b, :, fg_indices] if fg_indices.numel() > 0 else tracks.new_zeros((T, 0, 2))
            bg_track = tracks[b, :, bg_indices] if bg_indices.numel() > 0 else tracks.new_zeros((T, 0, 2))
            
            foreground_tracks.append(fg_track)
            background_tracks.append(bg_track)
        
        return foreground_tracks, background_tracks
    
    def random_k(self, n_points, max_k=5):
        """
        Randomly select number of clusters between 1 and min(max_k, n_points).
        
        Args:
            n_points: Number of points available
            max_k: Maximum number of clusters
        Returns:
            Number of clusters k
        """
        return np.random.randint(1, min(max_k, n_points) + 1)
    
    def cluster_traces(self, tracks, k):
        """
        Cluster traces using K-means.
        
        Args:
            tracks: Tensor of shape (T, N, 2) where T is time steps,
                   N is number of points, and 2 is (x,y) coordinates
            k: Number of clusters
        Returns:
            Cluster assignments and cluster centers
        """
        # Check if we have enough points to cluster
        if k <= 0 or tracks is None:
            return None, None
            
        # Get shape information
        if isinstance(tracks, list):
            # If it's a list, we need to handle it differently
            if len(tracks) == 0:
                return None, None
            tracks = tracks[0]  # Take the first element if it's a list
            
        # Check dimensions
        if len(tracks.shape) == 3:  # (T, N, 2)
            T, N, D = tracks.shape
            # Reshape tracks for clustering: each track becomes a feature vector
            flat_tracks = tracks.reshape(N, T*D).cpu().numpy()
        elif len(tracks.shape) == 4:  # (B, T, N, 2)
            B, T, N, D = tracks.shape
            # Reshape tracks for clustering
            flat_tracks = tracks.reshape(B*N, T*D).cpu().numpy()
        else:
            print(f"Unexpected shape for tracks: {tracks.shape}")
            return None, None
            
        # Make sure we have enough points to cluster
        if flat_tracks.shape[0] < k:
            k = max(1, flat_tracks.shape[0])
            
        # Apply K-means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(flat_tracks)
        centers = kmeans.cluster_centers_
        
        return clusters, centers
    
    def select_representative_points(self, tracks, clusters, n_per_cluster=1):
        """
        Select representative points from each cluster.
        
        Args:
            tracks: Tensor of shape (T, N, 2)
            clusters: Cluster assignments for each point
            n_per_cluster: Number of points to select per cluster
        Returns:
            Selected point indices
        """
        if clusters is None:
            return []
        
        # Get unique clusters
        unique_clusters = np.unique(clusters)
        selected_indices = []
        
        for cluster_id in unique_clusters:
            # Get indices of points in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            # Randomly select n_per_cluster points from this cluster
            if len(cluster_indices) > 0:
                n_select = min(n_per_cluster, len(cluster_indices))
                selected = np.random.choice(cluster_indices, size=n_select, replace=False)
                selected_indices.extend(selected)
        
        return selected_indices
    
    def apply_som_to_frame(self, frame, foreground_points, background_points=None):
        """
        Apply SoM to the first frame using foreground and background points.
        
        Args:
            frame: PIL Image of the first frame
            foreground_points: List of foreground point coordinates (x, y)
            background_points: Optional list of background point coordinates
        
        Returns:
            PIL Image with SoM applied
        """
        # Convert frame to PIL if it's not already
        if not isinstance(frame, Image.Image):
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            else:
                raise TypeError("Frame must be a PIL Image or numpy array")
        
        # Create a mask for the points
        mask = np.zeros((frame.height, frame.width), dtype=np.uint8)
        
        # Mark foreground points
        if foreground_points is not None and len(foreground_points) > 0:
            for point in foreground_points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < frame.width and 0 <= y < frame.height:
                    cv2.circle(mask, (x, y), 5, 255, -1)
        
        # Convert mask to PIL
        mask_pil = Image.fromarray(mask)
        
        # Create a dict with the image and mask as expected by SoM
        image_dict = {
            'background': frame,
            'layers': [mask_pil] if np.any(mask) else []
        }
        
        # Apply SoM using semantic-sam model
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # Use semantic-sam with level 3 (medium granularity)
                level = [3]
                text, text_part, text_thresh = '', '', '0.0'
                text_size, hole_scale, island_scale = 640, 100, 100
                semantic = False
                label_mode = '1'  # Use numbers for labels
                alpha = 0.2  # Transparency of masks
                anno_mode = ['Mask', 'Mark']  # Show both masks and marks
                
                output, mask = inference_semsam_m2m_auto(
                    self.model_semsam, 
                    frame, 
                    level, 
                    text, 
                    text_part, 
                    text_thresh, 
                    text_size, 
                    hole_scale, 
                    island_scale, 
                    semantic, 
                    label_mode=label_mode, 
                    alpha=alpha, 
                    anno_mode=anno_mode
                )
                
                return output
    
    def extract_first_frame(self, video_path):
        """
        Extract the first frame from a video.
        
        Args:
            video_path: Path to the video file
        
        Returns:
            PIL Image of the first frame
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        else:
            raise ValueError("Could not read the first frame from the video")
    
    def process_video(self, video_path, output_dir="output"):
        """
        Process video to generate SoM and ToM.
        
        Args:
            video_path: Path to input video
            output_dir: Output directory
        Returns:
            Processed tracks and visualization
        """
        # Step 1: Get initial tracks using CoTracker
        process_video_with_cotracker(
            video_path=video_path,
            output_dir=output_dir,
            grid_size=self.grid_size
        )
        
        # Load saved tracks
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        trace_path = f"{output_dir}/traces/{video_name}_traces.pt"
        print(f"Loading tracks from {trace_path}")
        
        data = torch.load(trace_path)
        tracks = data['tracks']
        visibility = data['visibility']
        
        print(f"Tracks shape: {tracks.shape}")
        print(f"Visibility shape: {visibility.shape}")
        
        # Step 2: Check and remove global motion
        has_global = self.has_global_motion(tracks, visibility)
        print(f"Has global motion: {has_global}")
        
        if has_global:
            print("Removing global motion...")
            tracks = remove_global_motion(tracks, visibility)
        
        # Step 3: Classify traces
        print("Classifying traces...")
        foreground_tracks, background_tracks = self.classify_traces(tracks, visibility)
        
        # Step 4-7: Cluster traces and select representatives
        print("Clustering traces...")
        fg_clusters_list = []
        bg_clusters_list = []
        fg_selected_points = []
        bg_selected_points = []
        
        for b in range(len(foreground_tracks)):
            fg_track = foreground_tracks[b]
            bg_track = background_tracks[b]
            
            print(f"Foreground track shape: {fg_track.shape}")
            print(f"Background track shape: {bg_track.shape}")
            
            # Step 6: Randomly select k for clustering
            if fg_track.shape[1] > 0:
                # Random k between 1 and min(5, |M^f|)
                k_fg = self.random_k(fg_track.shape[1], 5)
                print(f"Using k={k_fg} for foreground clustering")
                
                # Step 7: Apply K-means to foreground tracks
                fg_clusters, fg_centers = self.cluster_traces(fg_track, k_fg)
                fg_clusters_list.append(fg_clusters)
                
                # Select representative points from each cluster
                fg_selected = self.select_representative_points(fg_track, fg_clusters)
                fg_selected_points.append(fg_selected)
            else:
                print("No foreground points to cluster")
                fg_clusters_list.append(None)
                fg_selected_points.append([])
            
            if bg_track.shape[1] > 0:
                # Use 2k for background clustering
                k_bg = 2 * k_fg if 'k_fg' in locals() and k_fg > 0 else 2
                print(f"Using k={k_bg} for background clustering")
                
                # Apply K-means to background tracks
                bg_clusters, bg_centers = self.cluster_traces(bg_track, k_bg)
                bg_clusters_list.append(bg_clusters)
                
                # Select representative points from each cluster
                bg_selected = self.select_representative_points(bg_track, bg_clusters)
                bg_selected_points.append(bg_selected)
            else:
                print("No background points to cluster")
                bg_clusters_list.append(None)
                bg_selected_points.append([])
        
        # Step 8: Apply SoM to the first frame
        print("Applying SoM to the first frame...")
        first_frame = self.extract_first_frame(video_path)
        
        # Get foreground points from the first frame (t=0)
        fg_points = []
        for b in range(len(foreground_tracks)):
            if len(fg_selected_points[b]) > 0:
                # Get the coordinates at t=0 for selected points
                for idx in fg_selected_points[b]:
                    point = foreground_tracks[b][0, idx].cpu().numpy()  # t=0, selected index
                    fg_points.append(point)
        
        # Get background points from the first frame (t=0)
        bg_points = []
        for b in range(len(background_tracks)):
            if len(bg_selected_points[b]) > 0:
                # Get the coordinates at t=0 for selected points
                for idx in bg_selected_points[b]:
                    point = background_tracks[b][0, idx].cpu().numpy()  # t=0, selected index
                    bg_points.append(point)
        
        # Apply SoM to the first frame
        som_frame = self.apply_som_to_frame(first_frame, fg_points, bg_points)
        
        # Save the SoM frame
        som_path = os.path.join(output_dir, "som", f"{video_name}_som.png")
        os.makedirs(os.path.dirname(som_path), exist_ok=True)

        # Convert to PIL Image if it's a numpy array
        if isinstance(som_frame, np.ndarray):
            som_frame_pil = Image.fromarray(som_frame)
            som_frame_pil.save(som_path)
        else:
            som_frame.save(som_path)
        print(f"SoM frame saved to {som_path}")
        
        # Save results in a more compatible format
        result_path = os.path.join(output_dir, "results", f"{video_name}_tom.pt")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        # Convert tensors to numpy arrays for better compatibility
        torch.save({
            'tracks': tracks.cpu().numpy(),
            'foreground_tracks': [ft.cpu().numpy() for ft in foreground_tracks],
            'background_tracks': [bt.cpu().numpy() for bt in background_tracks],
            'foreground_clusters': fg_clusters_list,
            'background_clusters': bg_clusters_list,
            'foreground_selected': fg_selected_points,
            'background_selected': bg_selected_points,
            'som_path': som_path
        }, result_path, _use_new_zipfile_serialization=True)
        
        print(f"Results saved to {result_path}")
        
        return tracks, foreground_tracks, som_frame

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate SoM and ToM from video")
    parser.add_argument("--video", default="input.mp4", help="Path to input video")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--grid-size", type=int, default=15, 
                       help="Grid size for tracking (s=15)")
    parser.add_argument("--global-motion-threshold", type=float, default=2.0,
                       help="Threshold for global motion detection (η=2)")
    parser.add_argument("--foreground-threshold", type=float, default=2.0,
                       help="Threshold for foreground classification (ϵ=2)")
    
    args = parser.parse_args()
    
    generator = ToMGenerator(
        grid_size=args.grid_size,
        global_motion_threshold=args.global_motion_threshold,
        foreground_threshold=args.foreground_threshold
    )
    
    generator.process_video(args.video, args.output)

if __name__ == "__main__":
    main() 