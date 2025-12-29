import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def calculate_ssim_score(image1, image2, threshold=0.82):
    gray_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(gray_1, gray_2)
    #print(ssim_score)
    #return ssim_score > threshold
    return ssim_score

# Get sharpness score by computing laplasian variance

def get_sharpness_score(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray_image, cv2.CV_64F).var() 
    # 0-50 Blurry image, 50-300 Normal sharpness 300+ Sharp 
    return score

def calculate_frame_metrics(frame:np.array, last_frame, back_sub):
    """Calculates  motion ratio score, frame similarity SSIM score, and sharpness score """
    mask = back_sub.apply(frame)
    nonzero_count = cv2.countNonZero(mask)
    motion_ratio_score = nonzero_count / (360*640)

    # Use methods for ssim and sharpness scores
    ssim_score = calculate_ssim_score(frame, last_frame)
    sharpness_score = get_sharpness_score(frame)

    return (motion_ratio_score, ssim_score, sharpness_score)
    

def normalize(array):
    array = np.array(array)
    return (array - array.min()) / (array.max() - array.min())

def calculate_sharpness_motion(frame, last_frame, back_sub):
    mask = back_sub.apply(frame)
    nonzero_count = cv2.countNonZero(mask)
    motion_ratio_score = nonzero_count / (360*640)
    sharpness_score = get_sharpness_score(frame)

    return (motion_ratio_score, sharpness_score)

def calculate_weighted_score(motion_scores, sharpness_scores, w1=0.3, w2=0.7):
    return motion_scores*w1 + sharpness_scores*w2


def calculate_video_scores(video_path):
    """Calculates motion, ssim and sharpness score for each videoframe"""
    
    print("Calculate video scores")
    scores_dict = {"motion_scores": [],
                   "ssim_scores": [],
                   "sharpness_scores": []}

    cap = cv2.VideoCapture(video_path)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
    last_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        # 1. Resize for faster detection 
        small_frame = cv2.resize(frame, (640, 360))
        if last_frame is None:
          last_frame = small_frame
          continue

        if frame_count % 5 != 0:
            frame_count += 1
            continue
        
        motion_score, sharpness_score = calculate_sharpness_motion(small_frame, 
                                                                            last_frame,
                                                                            back_sub)
        # Add the scores for the crrent videoframe to the result diSct
        scores_dict["motion_scores"].append(motion_score)
        #scores_dict["ssim_scores"].append(ssim_score)
        scores_dict["sharpness_scores"].append(sharpness_score)
    
    # Normalize scores dict
    scores_dict["motion_scores"] = normalize(scores_dict["motion_scores"])
    #scores_dict["ssim_scores"]  = normalize(scores_dict["ssim_scores"])
    scores_dict["sharpness_scores"] = normalize(scores_dict["sharpness_scores"])
    scores_dict["weighted_scores"] = calculate_weighted_score(scores_dict["motion_scores"], 
                                                              scores_dict["sharpness_scores"])


    cap.release()
    print("All metrics saved successflly")
    
    return scores_dict

def get_scores_threshold(weighted_scores):
    return np.percentile(weighted_scores, 90)

colibri_frames_dir = "detected_frames/colibri"
koala_frames_dir = "detected_frames/koala"


def process_video_stream(video_path, output_folder="detected_frames/test"):
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

        
        if scores_dict["weighted_scores"][frame_idx] > scores_threshold: 
            if calculate_ssim_score(small_frame, last_frame) < 0.9:
                file_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(file_path, frame) # Save the original high-quality frame
                last_frame = small_frame
                saved_frames.append(file_path)
                saved_count += 1
                print(f"Frame {saved_count} saved")
            
        frame_idx += 1
        
        # Periodically clear output
        if frame_idx % 500 == 0:
            print(f"Processed {frame_idx} frames... saved {saved_count}")

    cap.release()
    print(f"Done! Check the {output_folder} folder.")
    return saved_frames