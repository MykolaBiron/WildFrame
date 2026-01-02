import streamlit as st
import os
import cv2
import tempfile
from pipeline_utils import *
from ultralytics import YOLO
import torch
from yolo_utils import classify_animal
from image_enhancer import enhance_image

# Import your custom modules
# from wildframe_logic import QualityAnalyzer, MotionDetector

# --- CONFIGURATION ---
st.set_page_config(page_title="WildFrame AI", layout="centered")

# 1. Initialize session state keys at the top of your app
if 'stills' not in st.session_state:
    st.session_state.stills = []
if 'detected_animals' not in st.session_state:
    st.session_state.detected_animals = []
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

extract_clicked = False

# --- CACHED LOGIC (Prevents reloading models on every click) ---
@st.cache_resource
def load_yolo():
    # Load YOLO or your quality scoring weights here
    return YOLO("yolov8s.pt")  # Will auto-download on first run

@st.cache_resource
def load_yolo_animals():
    return YOLO("ml_models/yolo_animals.pt")

@st.cache_resource
def load_rsgan():
    return torch.load("ml_models/ersgan.pth")

yolo = load_yolo()

# --- CORE PROCESSING FUNCTION ---
def process_video_stream(video_path, output_folder, n_frames):
    """Process  uploaded video and extract best frames based on weighted score"""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    # Create background subtractor object
    scores_dict = calculate_video_scores(video_path)
    
    saved_frames = []
    frame_idx = 0
    saved_count = 0
    last_frame = None
    scores_threshold = get_scores_threshold(scores_dict["weighted_scores"])

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

        results = yolo.predict(small_frame)
        if len(results[0].boxes) == 0:
            #Skip if frame does not contain an animal
            continue
        
        if scores_dict["weighted_scores"][frame_idx] > scores_threshold: 
            if calculate_ssim_score(small_frame, last_frame) < 0.9:
                file_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(file_path, frame) # Save the original high-quality frame
                last_frame = small_frame
                saved_frames.append(file_path)
                saved_count += 1
                print(f"Frame {saved_count} saved")

                # Get yolo classification
                st.session_state.detected_animals.append(classify_animal(yolo, file_path))
            
        frame_idx += 1
        
        # Periodically clear output
        if frame_idx % 500 == 0:
            print(f"Processed {frame_idx} frames... saved {saved_count}")

    cap.release()
    print(f"Done! Check the {output_folder} folder.")
    return saved_frames[:n_frames]


# --- USER INTERFACE ---
st.title("🐾 WildFrame: Wildlife Still Extractor")
st.write("Automatically capture the best high-quality stills from your videos.")

uploaded_file = st.file_uploader("Upload a wildlife video", type=['mp4', 'mov', 'avi'])
num_images = st.slider("Number of images to extract", 1, 10, 5)

enhanced_image = None
original_image = None

if uploaded_file is not None:
    if st.button("Extract Best Moments"):
        # Clear previous results
        st.session_state.stills = []
        st.session_state.enhanced_image = None
        st.session_state.original_image = None
        
        with st.spinner("AI is analyzing frames..."):
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            st.session_state.stills = process_video_stream(video_path, "detected_frames/test", num_images)
            extract_clicked = True

if st.session_state.stills:
    # Display results in a grid
    cols = st.columns(3)
    for idx, img in enumerate(st.session_state.stills):
        cols[idx % 3].image(img, caption=f"Best Shot #{idx+1}: {st.session_state.detected_animals[idx]}")
        if cols[idx % 3].button("Enhance quality", key=f"enhanced_{idx}"):
            # Run deep learning model
            original_image = img
            st.session_state.enhanced_image = enhance_image(img)
            st.session_state.original_image = img

else:
    if uploaded_file and st.session_state.get('stills') is not None and not extract_clicked:
        st.warning("⚠️ Unfortunately we were not able to detect any animals in your video...")

# Show the original and upscaled image to the user
if st.session_state.enhanced_image:
    st.divider()
    st.subheader("✨ High Quality Comparison")
    col_left, col_right = st.columns(2)
    col_left.image(st.session_state.original_image, caption="Original")
    col_right.image(st.session_state.enhanced_image, caption="AI Enhanced") 