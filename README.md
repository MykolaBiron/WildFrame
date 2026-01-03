# WildFrame Computer Vision app

Live Demo: https://wildframe-dpejshiibdrz4figxrfdp3.streamlit.app/

## 📦 Technologies
- Pytorch: Image upscaling deep learning model
- Ultralytics YOLOv8s: Real-time object detection and classification of 100 animal classes.
- Streamlit:	Responsive web UI for video uploads and still extraction.
- OpenCV	Frame sampling, Laplacian Sharpness scoring, and Motion estimation.
- Autodistill	Automated dataset annotation using Grounding DINO as a teacher model.
- Google Colab: Compute	NVIDIA T4 GPU	Cloud-based training environment 

## 🚀Features
- **High quality frames extraction from video**
- **Animal detection and classification**
- **Image upscaling using AI**

## 👩🏽‍🍳 The Process
The WildFrame AI system is divided into two major components: a Frame Scoring Engine for intelligent video sampling and a Custom Wildlife Detection Model for species identification.
1. Intelligent Frame Scoring & SelectionTo extract the highest quality stills while minimizing redundancy, I implemented a multi-stage ranking engine.Frame Sampling: Videos are processed using OpenCV to sample frames at a fixed interval, balancing computational speed with coverage.
2. The Scoring Engine: Each sampled frame is evaluated against three primary metrics:Motion Ratio: Identifies dynamic moments by calculating pixel displacement between sequential frames.Sharpness Score: Uses the Laplacian variance method to measure edge density, ensuring only in-focus shots are prioritized.SSIM (Structural Similarity Index): Once the top 10 sharpest/high-motion candidates are found, I apply SSIM to compare them against one another. Frames with a similarity score $> 0.9$ are flagged as duplicates and removed.Why this matters:
Without the SSIM deduplication step, a user might receive 5 nearly identical photos of a stationary animal. This logic ensures the final gallery is diverse and aesthetically varied.
3. Data Engineering with "Auto-Labeling"Training a model to recognize 100 different animal classes requires a massive, high-quality dataset. I used a "Distillation" approach to avoid weeks of manual labeling.Bulk Acquisition: I utilized the Pexels API to programmatically download 20,000 images (200 images per class for 100 species).Knowledge Distillation (Autodistill): * Manually labeling 20,000 images is impractical. Instead, I used Autodistill (Roboflow) to "distill" the knowledge of a large foundation model (Grounding DINO) into my smaller target model.I defined a Caption Ontology mapping text prompts (e.g., "a photo of a red fox") to my specific YOLO classes.The base model automatically generated bounding boxes and YOLOv8-formatted .txt annotations for the entire dataset.3. Model Fine-TuningThe final model is a fine-tuned version of Ultralytics YOLOv8s, chosen for its optimal balance between inference speed and detection accuracy.Training Environment: Conducted on Google Colab using NVIDIA T4 GPUs.
4. Training Specs: The process lasted approximately 10 hours, utilizing early stopping to prevent overfitting and a mosaic data augmentation strategy to help the model identify animals in complex, cluttered wildlife environments.
Optimization: The resulting weights were exported for deployment, capable of processing video streams in near real-time.

## 🍿 Video
