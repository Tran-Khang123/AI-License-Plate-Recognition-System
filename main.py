import os
import warnings
import threading
import asyncio
from contextlib import asynccontextmanager
import torch
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.routers import home, video
from config import plate_model_path, char_model_path

warnings.filterwarnings('ignore')

# Global vars
device = None
plate_model = None
char_model = None
ocr = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    global device, plate_model, char_model, ocr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    plate_model = None
    char_model = None
    try:
        if os.path.exists(plate_model_path):
            plate_model = YOLO(plate_model_path)
            print("Loaded plate model.")
        else:
            print("Plate model not found at", plate_model_path)
        if os.path.exists(char_model_path):
            char_model = YOLO(char_model_path)
            print("Loaded char model.")
        else:
            print("Char model not found at", char_model_path)
    except Exception as e:
        print("Model load warning:", e)
        plate_model = None
        char_model = None

    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        print("PaddleOCR ready.")
    except Exception as e:
        print("PaddleOCR init failed:", e)
        ocr = None

    loop = asyncio.get_event_loop()
    t = threading.Thread(target=video.frames_worker, args=(loop,), daemon=True)
    t.start()
    asyncio.create_task(video.broadcaster_task())
    print("Server background tasks started.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(home.router)
app.include_router(video.router)

app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}/"
    print("=====================================")
    print("🚀 Server đang chạy, mở trình duyệt tại:")
    print(f"👉 {url}")
    print("=====================================")
    uvicorn.run("main:app", host=host, port=port, reload=True)