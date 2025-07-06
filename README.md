
<a href="[https://your-destination-link.com](https://www.linkedin.com/feed/update/urn:li:ugcPost:7345461036101124096)">
  <img src="thumbnail" alt="Thumbnail" width="80%">
</a>

[View this video post on LinkedIn](https://www.linkedin.com/feed/update/urn:li:ugcPost:7345461036101124096)

# 🚨 Gun Detection and Alerting with YOLOv8 and Twilio

A real-time gun detection system using a YOLOv8 model fine-tuned on the Armas dataset. This project provides tools for detecting firearms in images, videos, and live webcam feeds, with optional real-time alerts via Twilio (call/SMS).

## Features
- **Image Detection**: Upload an image and detect firearms using a Gradio web interface.
- **Video Detection**: Run detection on video files and save annotated results.
- **Live Feed Detection**: Monitor a webcam feed for firearms, save detection frames, and send real-time alerts via Twilio.
- **Custom Training**: Train your own YOLOv8 model on the provided or custom dataset.

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Gun_Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Image Detection (Gradio Web App)
```bash
python app.py
```
- Open the provided local URL in your browser.
- Upload an image to see detection results.

### 2. Video Detection
Place your video in the `videos/` directory (e.g., `input_2.mp4`), then run:
```bash
python predict_video.py
```
- Output will be saved as `videos/input_2_out.mp4`.

### 3. Live Feed Detection with Alerts
```bash
python live_feed.py
```
- Requires a webcam.
- Configure your Twilio credentials in `live_feed.py` for call/SMS alerts.
- Detection frames are saved in the `detections/` directory.

### 4. Model Training
```bash
python train.py
```
- Trains a YOLOv8 model using the configuration in `config.yaml`.

## Dataset
- Dataset path: `data/images/`
- Configured in `config.yaml` (uses the Armas dataset from Roboflow)
- Class: `gun`
- [Roboflow Dataset Link](https://universe.roboflow.com/joseph-nelson/pistols/dataset/1)

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Acknowledgments
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)
- [Twilio](https://www.twilio.com/)

---
*For questions or contributions, please open an issue or pull request.* 
