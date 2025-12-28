import streamlit as st
import os
import cv2
import tempfile
from pipeline_utils import *


# Import your custom modules
# from wildframe_logic import QualityAnalyzer, MotionDetector

# --- CONFIGURATION ---
st.set_page_config(page_title="WildFrame AI", layout="centered")

# --- CACHED LOGIC (Prevents reloading models on every click) ---
@st.cache_resource
def load_models():
    # Load YOLO or your quality scoring weights here
    return None

# --- CORE PROCESSING FUNCTION ---
def process_video_stream(video_path, output_folder="detected_frames"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    # Create background subtractor object
    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
    #scores_dict = calculate_video_scores("videos/colibri_video1.mp4")
    
    frame_idx = 0
    saved_count = 0
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Process only every 5th frame
        if frame_idx % 5 != 0:
            frame_idx += 1
            continue
        
        # 1. Resize for detection 
        small_frame = cv2.resize(frame, (640, 360))
        
        # 3. Save to disk if motion is found (instead of appending to a list)
        if last_frame is None:
          last_frame = small_frame
          frame_idx += 1
          continue

     
        motion_score, ssim_score, sharpness_score = calculate_frame_metrics(small_frame, last_frame, back_sub)
        
        if calculate_weighted_score(scores_dict, frame_idx // 5) > pick_best_frames(scores_dict): # 0.3% motion and 700 mse threshold
            file_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(file_path, frame) # Save the original high-quality frame
            last_frame = small_frame
            saved_count += 1
            print(f"Weighted score: {calculate_weighted_score(scores_dict, frame_idx)}")
            print(f"Frame {saved_count} saved")
            
        frame_idx += 1
        
        # Periodically clear output
        if frame_idx % 500 == 0:
            print(f"Processed {frame_idx} frames... saved {saved_count}")

    cap.release()
    print(f"Done! Check the {output_folder} folder.")

# --- USER INTERFACE ---
st.title("🐾 WildFrame: Wildlife Still Extractor")
st.write("Automatically capture the best high-quality stills from your videos.")

uploaded_file = st.file_uploader("Upload a wildlife video", type=['mp4', 'mov', 'avi'])
num_images = st.slider("Number of images to extract", 1, 10, 5)

if uploaded_file is not None:
    if st.button("Extract Best Moments"):
        with st.spinner("AI is analyzing frames..."):
            stills = process_video_stream(uploaded_file, num_images)
            
            # Display results in a grid
            cols = st.columns(3)
            for idx, img in enumerate(stills):
                cols[idx % 3].image(img, caption=f"Best Shot #{idx+1}")