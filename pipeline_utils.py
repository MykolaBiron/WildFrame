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


    cap.release()
    print("All metrics saved successflly")
    
    return scores_dict
        
scores_dict = calculate_video_scores("videos/colibri_video1.mp4")