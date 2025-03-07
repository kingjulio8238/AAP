import os
import sys
import tempfile
from PIL import Image, ImageDraw

# Fix NumPy and OpenCV compatibility issues
os.system('pip install numpy==1.23.5')
os.system('pip install opencv-python==4.8.1.78')

# Patch the problematic cv2 typing
import builtins
_orig_import = builtins.__import__

def _patched_import(name, *args, **kwargs):
    if name == 'cv2.typing':
        # Skip importing cv2.typing which causes the error
        import cv2
        return cv2
    return _orig_import(name, *args, **kwargs)

builtins.__import__ = _patched_import

# Now import the rest
import numpy as np
import gradio as gr
import torch
import cv2
from tom import ToMGenerator

def process_video_with_tom(
    video_file, 
    grid_size=15, 
    global_motion_threshold=2.0, 
    foreground_threshold=2.0,
    progress=gr.Progress()
):
    """
    Process a video with ToM Generator and return results for visualization
    """
    try:
        # Create a temporary directory for outputs
        temp_dir = tempfile.mkdtemp()
        
        # Save the uploaded video to the temp directory
        video_path = os.path.join(temp_dir, "input_video.mp4")
        
        # Handle different types of video input
        if isinstance(video_file, str):
            # If it's a string (file path), copy the file
            import shutil
            shutil.copy(video_file, video_path)
        else:
            # If it's bytes (uploaded file), write the bytes
            with open(video_path, "wb") as f:
                f.write(video_file)
        
        # Initialize the ToM Generator
        progress(0.1, "Initializing ToM Generator...")
        generator = ToMGenerator(
            grid_size=grid_size,
            global_motion_threshold=global_motion_threshold,
            foreground_threshold=foreground_threshold
        )
        
        # Process the video
        progress(0.2, "Processing video with CoTracker...")
        tracks, foreground_tracks, som_frame = generator.process_video(video_path, temp_dir)
        
        # Extract the first frame for comparison
        progress(0.9, "Preparing results...")
        first_frame = generator.extract_first_frame(video_path)
        
        # Get the SoM path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        som_path = os.path.join(temp_dir, "som", f"{video_name}_som.png")
        
        # Load the result data with weights_only=False to avoid unpickling error
        result_path = os.path.join(temp_dir, "results", f"{video_name}_tom.pt")
        try:
            # First try with default settings
            result_data = torch.load(result_path)
        except Exception as e:
            print(f"Error loading with default settings: {e}")
            # If that fails, try with weights_only=False
            result_data = torch.load(result_path, weights_only=False)
            print("Successfully loaded with weights_only=False")
        
        # Create a visualization of the tracks
        fg_tracks = result_data['foreground_tracks'][0]  # First batch
        bg_tracks = result_data['background_tracks'][0]  # First batch
        
        # Convert first frame to numpy for visualization
        first_frame_np = np.array(first_frame)
        
        # Create a visualization of the tracks on the first frame
        track_vis = first_frame_np.copy()
        
        # Draw foreground tracks as red points
        if isinstance(fg_tracks, np.ndarray) and fg_tracks.shape[1] > 0:
            for i in range(fg_tracks.shape[1]):
                x, y = fg_tracks[0, i]  # t=0, point i
                x, y = int(x), int(y)
                if 0 <= x < track_vis.shape[1] and 0 <= y < track_vis.shape[0]:
                    cv2.circle(track_vis, (x, y), 3, (255, 0, 0), -1)  # Red for foreground
        elif isinstance(fg_tracks, torch.Tensor) and fg_tracks.shape[1] > 0:
            for i in range(fg_tracks.shape[1]):
                x, y = fg_tracks[0, i].cpu().numpy()  # t=0, point i
                x, y = int(x), int(y)
                if 0 <= x < track_vis.shape[1] and 0 <= y < track_vis.shape[0]:
                    cv2.circle(track_vis, (x, y), 3, (255, 0, 0), -1)  # Red for foreground
        
        # Draw background tracks as blue points
        if isinstance(bg_tracks, np.ndarray) and bg_tracks.shape[1] > 0:
            for i in range(bg_tracks.shape[1]):
                x, y = bg_tracks[0, i]  # t=0, point i
                x, y = int(x), int(y)
                if 0 <= x < track_vis.shape[1] and 0 <= y < track_vis.shape[0]:
                    cv2.circle(track_vis, (x, y), 3, (0, 0, 255), -1)  # Blue for background
        elif isinstance(bg_tracks, torch.Tensor) and bg_tracks.shape[1] > 0:
            for i in range(bg_tracks.shape[1]):
                x, y = bg_tracks[0, i].cpu().numpy()  # t=0, point i
                x, y = int(x), int(y)
                if 0 <= x < track_vis.shape[1] and 0 <= y < track_vis.shape[0]:
                    cv2.circle(track_vis, (x, y), 3, (0, 0, 255), -1)  # Blue for background
        
        # Convert track visualization to PIL
        track_vis_pil = Image.fromarray(track_vis)
        
        # Ensure som_frame is a PIL Image
        if isinstance(som_frame, np.ndarray):
            som_frame = Image.fromarray(som_frame)
        
        progress(1.0, "Done!")
        return first_frame, track_vis_pil, som_frame
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # Create error images
        error_img = Image.new('RGB', (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(error_img)
        draw.text((10, 10), f"Error processing video:\n{str(e)}", fill=(255, 0, 0))
        
        return error_img, error_img, error_img

# Create the Gradio interface
with gr.Blocks(title="ToM Generator App") as app:
    gr.Markdown("# ToM Generator: Set-of-Mark (SoM) and Traces-of-Mark (ToM) for Videos")
    gr.Markdown("""
    This app applies the ToM algorithm to videos, which:
    1. Tracks points using CoTracker
    2. Removes global motion if present
    3. Classifies traces into foreground and background
    4. Clusters traces and selects representatives
    5. Applies SoM to the first frame
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            
            with gr.Row():
                grid_size = gr.Slider(minimum=5, maximum=30, value=15, step=1, 
                                     label="Grid Size (s)", info="Size of tracking grid")
                global_motion_threshold = gr.Slider(minimum=0.5, maximum=5.0, value=2.0, step=0.1,
                                                  label="Global Motion Threshold (η)", 
                                                  info="Threshold for global motion detection")
                foreground_threshold = gr.Slider(minimum=0.5, maximum=5.0, value=2.0, step=0.1,
                                               label="Foreground Threshold (ϵ)",
                                               info="Threshold for foreground classification")
            
            process_btn = gr.Button("Process Video", variant="primary")
        
    with gr.Row():
        with gr.Column():
            original_frame = gr.Image(label="First Frame", type="pil")
        with gr.Column():
            tracks_vis = gr.Image(label="Tracks Visualization", type="pil", 
                                info="Red: Foreground, Blue: Background")
        with gr.Column():
            som_frame = gr.Image(label="SoM Result", type="pil")
    
    # Set up the processing function
    process_btn.click(
        process_video_with_tom,
        inputs=[video_input, grid_size, global_motion_threshold, foreground_threshold],
        outputs=[original_frame, tracks_vis, som_frame]
    )
    
    # # Add examples
    # gr.Examples(
    #     examples=[
    #         ["examples/sample_video.mp4", 15, 2.0, 2.0],
    #     ],
    #     inputs=[video_input, grid_size, global_motion_threshold, foreground_threshold],
    # )

# Launch the app
if __name__ == "__main__":
    # Restore original import function before launching the app
    builtins.__import__ = _orig_import
    # Enable queue for progress tracking
    app.queue().launch(share=True, server_port=6093) 