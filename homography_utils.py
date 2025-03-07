import numpy as np
import cv2
import torch

def compute_homography(src_points, dst_points):
    """
    Compute homography matrix between source and destination points.
    
    Args:
        src_points: Source points (Nx2)
        dst_points: Destination points (Nx2)
    Returns:
        3x3 homography matrix
    """
    # Convert to numpy if tensor
    if isinstance(src_points, torch.Tensor):
        src_points = src_points.cpu().numpy()
    if isinstance(dst_points, torch.Tensor):
        dst_points = dst_points.cpu().numpy()
        
    # Ensure points are in correct format
    src_points = src_points.reshape(-1, 1, 2)
    dst_points = dst_points.reshape(-1, 1, 2)
    
    # Compute homography matrix
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H

def apply_homography_to_points(points, H):
    """
    Apply homography transformation to a set of points.
    
    Args:
        points: Points to transform (Nx2)
        H: 3x3 homography matrix
    Returns:
        Transformed points (Nx2)
    """
    # Convert to numpy if tensor
    is_tensor = isinstance(points, torch.Tensor)
    if is_tensor:
        points = points.cpu().numpy()
    
    # Reshape points to (N, 1, 2)
    points = points.reshape(-1, 1, 2)
    
    # Apply transformation
    transformed_points = cv2.perspectiveTransform(points, H)
    
    # Reshape back to (N, 2)
    transformed_points = transformed_points.reshape(-1, 2)
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        transformed_points = torch.from_numpy(transformed_points)
        
    return transformed_points

def remove_global_motion(tracks, visibility=None, threshold=0.1):
    """
    Remove global motion from tracks using homography transformation.
    
    Args:
        tracks: Tensor of shape (B, T, N, 2) where B is batch size, T is time steps,
               N is number of points, and 2 is (x,y) coordinates
        visibility: Optional tensor of shape (B, T, N) indicating point visibility
        threshold: Threshold for considering global motion
    Returns:
        Transformed tracks with global motion removed
    """
    B, T, N, _ = tracks.shape
    transformed_tracks = tracks.clone()
    
    for b in range(B):
        # Use first frame as reference
        ref_points = tracks[b, 0]  # (N, 2)
        
        for t in range(1, T):
            curr_points = tracks[b, t]  # (N, 2)
            
            # Filter points based on visibility if provided
            if visibility is not None:
                valid_mask = visibility[b, t] * visibility[b, 0] > 0.5
                if valid_mask.sum() < 4:  # Need at least 4 points for homography
                    continue
                    
                valid_ref = ref_points[valid_mask]
                valid_curr = curr_points[valid_mask]
            else:
                valid_ref = ref_points
                valid_curr = curr_points
            
            # Compute homography
            H = compute_homography(valid_curr, valid_ref)
            
            # Apply transformation to all points in current frame
            transformed_tracks[b, t] = apply_homography_to_points(curr_points, H)
    
    return transformed_tracks 