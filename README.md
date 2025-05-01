# 🚗 Imitation Learning in CARLA Simulator

This project demonstrates an imitation learning pipeline for autonomous driving using the CARLA simulator.  
A CNN model is trained on expert driving data (BehaviorAgent), and the trained agent is then deployed to control a car in the simulator.

---

## 🧠 Overview

- **Simulator**: [CARLA 0.10.0](https://carla.org/)
- **Task**: Predict steering, throttle, and brake from RGB camera input
- **Model**: Custom CNN (`EnhancedCNN`) with separate output heads
- **Data**: 224x224 RGB images + `[steer, throttle, brake]` from BehaviorAgent
- **Training**: MSE Loss on control outputs
- **Evaluation**: Inference in Town10 with model outputs logged and visualized

---

## 🎥 Inference Demo (CARLA)

[YouTube動画のサムネイル画像（クリック可能）]

[![Watch the video](https://img.youtube.com/vi/w7Hh2AnSmRg/0.jpg)](https://youtube.com/shorts/w7Hh2AnSmRg)

📁 Saved Video: `logs/output_video.avi`


---

## 📦 Project Structure
## 🔧 File Guide

| File / Folder                | Description                                |
|-----------------------------|--------------------------------------------|
| `models/`                   | Trained models (enhanced & baseline)       |
| `src/train_imitation1.py`   | Training script for imitation learning     |
| `src/record_data1.py`       | Collects expert driving data from CARLA    |
| `src/infer_imitation*.py`   | Different inference variants               |
| `src/make_video.py`         | Converts inference images into video       |
| `modules/enhanced_model.py` | CNN model definition (EnhancedCNN)         |
| `logs/`                     | Output logs and generated video            |
| `saved_data/`               | Collected dataset (images & actions)       |

## 🚀 Try It Yourself

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (optional)
python src/train_imitation1.py

# Run inference and save video
python src/infer_imitation_enhanced.py
python src/make_video.py

