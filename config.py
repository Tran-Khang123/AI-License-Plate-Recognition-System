import os

# Paths
plate_model_path = r"E:\GUI_DOANTOTNGHIEP\source_code_project\models\yolov11m_BienSo_V4.pt"
char_model_path  = r"E:\GUI_DOANTOTNGHIEP\source_code_project\models\YOLOv11l_OCR_256_V4.pt"
DEFAULT_VIDEO_PATH = r"E:\GUI_DOANTOTNGHIEP\video\IMG_0801.MOV"

# Directories
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FAILED_OCR_DIR = os.path.join(OUTPUT_DIR, "failed_ocr")
os.makedirs(FAILED_OCR_DIR, exist_ok=True)

# OCR
OCR_ATTEMPTS_NEEDED = 5
OCR_FRAME_INTERVAL = 9
MIN_VOTE_LENGTH = 6

# Payment
COST_PER_PASS = 20000