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

## 🎥 Inference Demo

https://user-uploaded-video-link-or-youtube.com

📁 Saved Video: `logs/output_video.avi`

---

## 📦 Project Structure

