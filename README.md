# 🚗 AI License Plate Recognition System

Real-time **license plate recognition and smart parking payment system** built with **YOLOv11, PaddleOCR, and FastAPI**.

This project detects vehicles from video streams, recognizes license plates, and automatically deducts parking fees from registered vehicles.

---

# ✨ Features

• Real-time license plate detection using YOLOv11  
• Character recognition using YOLO OCR model  
• Multi-frame OCR voting algorithm for higher accuracy  
• Real-time video streaming with MJPEG  
• WebSocket communication for instant detection results  
• Vehicle tracking using **BoT-SORT tracker**  
• ROI-based detection area control  
• Smart parking payment system  

---

# 🧠 AI Pipeline

Video Input  
    ↓  
YOLOv11 Plate Detection  
    ↓  
Plate Cropping  
    ↓  
Character Detection  
    ↓  
OCR Voting Algorithm  
    ↓  
License Plate Output  
    ↓  
Automatic Payment Deduction

---

# 🛠 Tech Stack

Backend
- FastAPI
- Uvicorn

AI / Computer Vision
- YOLOv11
- PaddleOCR
- OpenCV
- NumPy

Deep Learning
- PyTorch

Tracking
- BoT-SORT

Streaming
- WebSocket
- MJPEG

---

# 📂 Project Structure

```
AI-License-Plate-Recognition-System/
│
├── api/
│   └── routers/
│       ├── __init__.py
│       ├── home.py              # Home routes (UI rendering)
│       └── video.py             # Video processing endpoints
│
├── core/
│   ├── __init__.py
│   └── models.py                # Load AI models (YOLO, OCR)
│
├── services/
│   ├── __init__.py
│   ├── plate_processing.py      # License plate detection + OCR pipeline
│   └── payment.py               # Smart parking payment logic
│
├── models/
│   ├── __init__.py
│   └── botsort.yaml             # Tracking configuration (BoT-SORT)
│
├── templates/
│   └── GUI.html                 # Web interface
│
├── static/
│   └── style.css                # Frontend styling
│
├── config.py                    # System configuration
├── main.py                      # FastAPI application entrypoint
│
└── .gitignore
```

---

# 📡 API Endpoints

### Detect plate from image

```http
POST /process_image
```

### Upload video

```http
POST /upload_video
```

### Start video detection

```http
POST /start_video
```

### Register vehicle

```http
POST /register_car
```

### Delete vehicle

```http
DELETE /delete_car
```


---

# 🚀 Future Improvements

• Deploy with Docker  
• Use database (PostgreSQL / MongoDB)  
• Add vehicle classification  
• Optimize inference with TensorRT  

---

# 👨‍💻 Author

Tran Khang  
AI / Computer Vision Backend Project