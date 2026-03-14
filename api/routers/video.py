from fastapi import APIRouter, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import HTTPException
import asyncio
import threading
import time
import cv2
import numpy as np
import uuid
import os
import json
from collections import Counter
from config import OUTPUT_DIR, OCR_ATTEMPTS_NEEDED, OCR_FRAME_INTERVAL, FAILED_OCR_DIR, DEFAULT_VIDEO_PATH
from core.models import ROIModel
from services.payment import deduct_amount
import math
import base64
from typing import List, Dict, Optional
from config import MIN_VOTE_LENGTH
router = APIRouter()


FRAME_W = 1280
FRAME_H = 720
TARGET_FPS = 25
FPS = TARGET_FPS
video_file_lock = threading.Lock()
current_video_path = DEFAULT_VIDEO_PATH if os.path.exists(DEFAULT_VIDEO_PATH) else None
roi_box = [0, 0, FRAME_W, FRAME_H]
roi_lock = threading.Lock()
video_paused = True
video_control_lock = threading.Lock()
connected_websockets: List[WebSocket] = []
websockets_lock = asyncio.Lock()
latest_frame_bytes = None
frame_lock = threading.Lock()
track_states: Dict[int, dict] = {}
detection_queue: asyncio.Queue = asyncio.Queue()

def encode_jpg(img):
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

def safe_b64_from_img(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()

def is_center_in_roi(box, roi):
    bx1, by1, bx2, by2 = box
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    return (roi[0] <= cx <= roi[2]) and (roi[1] <= cy <= roi[3])

def vote_text_awcv(texts: List[str], confidences: Optional[List[List[float]]] = None) -> str:
    if not texts:
        return ""
    lengths = [len(t) for t in texts]
    most_common_len = Counter(lengths).most_common(1)[0][0]
    reference = next((t for t in texts if len(t) == most_common_len), texts[0])
    aligned_texts = []

    def align_pair(ref, target):
        m, n = len(ref), len(target)
        dp = np.zeros((m+1, n+1), dtype=int)
        for i in range(m+1):
            dp[i][0] = -i
        for j in range(n+1):
            dp[0][j] = -j
        for i in range(1, m+1):
            for j in range(1, n+1):
                match = dp[i-1][j-1] + (1 if ref[i-1] == target[j-1] else -1)
                dp[i][j] = max(match, dp[i-1][j]-1, dp[i][j-1]-1)
        aligned_ref, aligned_target = [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (1 if ref[i-1] == target[j-1] else -1):
                aligned_ref.append(ref[i-1])
                aligned_target.append(target[j-1])
                i, j = i-1, j-1
            elif i > 0 and dp[i][j] == dp[i-1][j]-1:
                aligned_ref.append(ref[i-1])
                aligned_target.append('-')
                i -= 1
            else:
                aligned_ref.append('-')
                aligned_target.append(target[j-1])
                j -= 1
        return ''.join(reversed(aligned_ref)), ''.join(reversed(aligned_target))

    for t in texts:
        _, aligned_t = align_pair(reference, t)
        aligned_texts.append(aligned_t)

    max_len = max(len(t) for t in aligned_texts)
    padded = [t.ljust(max_len, '-') for t in aligned_texts]
    voted_chars = []
    for pos in range(max_len):
        char_weights = {}
        for i, text in enumerate(padded):
            ch = text[pos]
            if ch == '-':
                continue
            weight = 1.0
            if confidences and i < len(confidences) and pos < len(confidences[i]):
                weight += confidences[i][pos]
            char_weights[ch] = char_weights.get(ch, 0) + weight
        if char_weights:
            best_char = max(char_weights, key=char_weights.get)
        else:
            best_char = ''
        voted_chars.append(best_char)
    return ''.join(voted_chars).replace('-', '')

def finalize_ocr_for_track(tid: int, state: dict, x1, y1, x2, y2, conf: float, frame_idx: int, loop):
    if not state['ocr_texts']:
        return
    voted_text = vote_text_awcv(state['ocr_texts'], state.get('ocr_confidences'))
    avg_conf = np.mean([np.mean(c) for c in state['ocr_confidences'] if c]) if state.get('ocr_confidences') else 0.0
    print(f"Finalized OCR for ID {tid}: '{voted_text}' (avg conf={avg_conf:.2f}) from {len(state['ocr_texts'])} attempts: {state['ocr_texts']}")

    id_folder = os.path.join(OUTPUT_DIR, f"track_id_{tid}")
    os.makedirs(id_folder, exist_ok=True)
    for i, crop_img in enumerate(state['ocr_crops']):
        crop_filename = f"frame_{state['ocr_attempts'][i]}_crop.jpg"
        crop_path = os.path.join(id_folder, crop_filename)
        cv2.imwrite(crop_path, crop_img)

    if len(voted_text) >= MIN_VOTE_LENGTH:
        payment = deduct_amount(voted_text)
        crop_b64_final = safe_b64_from_img(state['ocr_crops'][-1])
        from datetime import datetime
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        det = {
            "text": voted_text,
            "crop_b64": crop_b64_final,
            "frame_idx": int(frame_idx),
            "track_id": int(tid),
            "ts": time.time(),
            "payment": payment,
            "timestamp": current_time,
        }
        try:
            asyncio.run_coroutine_threadsafe(detection_queue.put(det), loop)
            print(f"Sent detection for ID {tid} to GUI: '{voted_text}'")
        except Exception as e:
            print("Failed to queue detection:", e)
    else:
        failed_id_folder = os.path.join(FAILED_OCR_DIR, f"track_id_{tid}_failed")
        os.makedirs(failed_id_folder, exist_ok=True)
        for i, crop_img in enumerate(state['ocr_crops']):
            crop_filename = f"frame_{state['ocr_attempts'][i]}_crop.jpg"
            crop_path = os.path.join(failed_id_folder, crop_filename)
            cv2.imwrite(crop_path, crop_img)
        print(f"Saved crops for failed ID {tid} to {failed_id_folder}")

    state['ocr_done'] = True


def frames_worker(loop):
    global latest_frame_bytes, track_states, detection_queue, FRAME_W, FRAME_H, FPS
    print("Starting frames_worker thread...")
    frame_idx = 0
    MIN_CONF = 0.40
    PLATE_IMGSZ = 736
    CHAR_IMGSZ = 256
    cap = None
    current_path_local = None

    while True:
        with video_control_lock:
            paused = video_paused
        with video_file_lock:
            global current_video_path
            if current_video_path != current_path_local:
                if cap is not None:
                    try: cap.release()
                    except: pass
                cap = None
                current_path_local = current_video_path
                if current_path_local and os.path.exists(current_path_local):
                    cap = cv2.VideoCapture(current_path_local)
                    if cap.isOpened():
                        FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_W
                        FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_H
                        print("Opened video:", current_path_local, FRAME_W, FRAME_H, FPS)
                else:
                    cap = None

        if paused:
            time.sleep(0.05)
            continue

        if cap is None:
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            ret = False
        else:
            start_read_ts = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        frame_idx += 1
        proc_frame = frame.copy()
        annotated = proc_frame.copy()
        results = []

        from main import plate_model, char_model, device  # Import từ main
        if plate_model is not None:
            try:
                results = plate_model.track(proc_frame, persist=True, tracker=r"E:\GUI_DOANTOTNGHIEP\source_code_project\models\botsort.yaml", device=device, imgsz=PLATE_IMGSZ, conf=MIN_CONF, verbose=False)
            except Exception as e:
                try:
                    results = plate_model(proc_frame, device=device, imgsz=PLATE_IMGSZ, conf=MIN_CONF)
                except Exception as e2:
                    print("Plate model video error:", e, e2)

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and getattr(result.boxes, "xyxy", None) is not None:
                try:
                    boxes_np = result.boxes.xyxy.cpu().numpy()
                except Exception:
                    try: boxes_np = result.boxes.xyxy.numpy()
                    except: boxes_np = []
                try:
                    confs_np = result.boxes.conf.cpu().numpy()
                except Exception:
                    try: confs_np = result.boxes.conf.numpy()
                    except: confs_np = []
                try:
                    ids_np = result.boxes.id.cpu().numpy().astype(int) if getattr(result.boxes, "id", None) is not None else None
                except Exception:
                    ids_np = None

                for i in range(len(boxes_np)):
                    try:
                        x1, y1, x2, y2 = map(int, boxes_np[i])
                        tid = int(ids_np[i]) if ids_np is not None else int(i)
                        conf = float(confs_np[i]) if len(confs_np) > i else 0.0
                    except Exception:
                        continue

                    w = x2 - x1; h = y2 - y1
                    if w < 30 or h < 20 or conf < MIN_CONF:
                        continue

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(annotated, f"ID {tid}", (x1, max(y1-6,6)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                    with roi_lock:
                        local_roi = roi_box.copy()
                    in_roi = is_center_in_roi((x1,y1,x2,y2), local_roi)

                    if tid not in track_states:
                        track_states[tid] = {
                            'status': 'out_roi',
                            'ocr_attempts': [],
                            'ocr_texts': [],
                            'ocr_crops': [],
                            'ocr_confidences': [],
                            'ocr_done': False,
                            'first_in_roi_frame': None,
                            'last_ocr_frame': None
                        }

                    state = track_states[tid]
                    current_status = 'in_roi' if in_roi else 'out_roi'
                    was_in_roi = track_states[tid]['status'] == 'in_roi'
                    track_states[tid]['status'] = current_status

                    should_ocr = (
                        in_roi and
                        not state['ocr_done'] and
                        len(state['ocr_texts']) < OCR_ATTEMPTS_NEEDED and
                        char_model is not None
                    )
                    if len(state['ocr_texts']) > 0:
                        last_ocr_frame = state.get('last_ocr_frame', -1000)
                        should_ocr = should_ocr and (frame_idx - last_ocr_frame >= OCR_FRAME_INTERVAL)

                    if should_ocr:
                        if state['first_in_roi_frame'] is None:
                            state['first_in_roi_frame'] = frame_idx

                        shrink = 2
                        cx1 = max(0, x1 + shrink)
                        cy1 = max(0, y1 + shrink)
                        cx2 = min(FRAME_W - 1, x2 - shrink)
                        cy2 = min(FRAME_H - 1, y2 - shrink)
                        plate_crop = frame[cy1:cy2, cx1:cx2].copy()
                        if plate_crop.size == 0:
                            continue

                        try:
                            chars_res = char_model(plate_crop, device=device, imgsz=CHAR_IMGSZ, conf=0.6, verbose=False)
                        except Exception as e:
                            print(f"char_model error (video) for ID {tid}: {e}")
                            chars_res = []

                        characters = []
                        for rc in chars_res:
                            boxes_char = getattr(rc, "boxes", None)
                            if boxes_char is None: continue
                            for bc in boxes_char:
                                try:
                                    vals = bc.xyxy[0].cpu().numpy().astype(int)
                                    xx1, yy1, xx2, yy2 = map(int, vals)
                                    cconf = float(bc.conf[0].cpu().numpy())
                                    cls = int(bc.cls[0].cpu().numpy())
                                    name = char_model.names[cls] if hasattr(char_model, 'names') else str(cls)
                                except Exception:
                                    continue
                                cx_ch = (xx1 + xx2) / 2.0; cy_ch = (yy1 + yy2) / 2.0
                                characters.append({'bbox':[xx1,yy1,xx2,yy2],'confidence':cconf,'class_name':str(name),'cx':cx_ch,'cy':cy_ch,'w':xx2-xx1,'h':yy2-yy1})

                        characters = [c for c in characters if c['confidence'] >= 0.65]

                        final_text = ""
                        final_conf = []
                        if characters:
                            heights = [c['h'] for c in characters if c['h']>0]
                            median_h = float(np.median(heights)) if heights else 20.0
                            row_thresh = max(8.0, median_h * 0.6)
                            chars_by_cy = sorted(characters, key=lambda x: x['cy'])
                            rows = []
                            for cch in chars_by_cy:
                                placed = False
                                for rrow in rows:
                                    if abs(cch['cy'] - rrow['mean_y']) <= row_thresh:
                                        rrow['items'].append(cch)
                                        rrow['mean_y'] = sum(it['cy'] for it in rrow['items'])/len(rrow['items'])
                                        placed=True
                                        break
                                if not placed:
                                    rows.append({'mean_y': cch['cy'], 'items':[cch]})

                            rows = sorted(rows, key=lambda r: r['mean_y'])
                            texts_per_row=[]
                            confs_per_row = []
                            for rrow in rows:
                                row_items = sorted(rrow['items'], key=lambda x: x['cx'])
                                row_text = ''.join([it['class_name'] for it in row_items])
                                row_conf = [it['confidence'] for it in row_items]
                                texts_per_row.append(row_text)
                                confs_per_row.append(row_conf)

                            final_text = ''.join(texts_per_row) if texts_per_row else ""
                            final_conf = [c for row in confs_per_row for c in row] if confs_per_row else []

                        state['ocr_texts'].append(final_text)
                        state['ocr_confidences'].append(final_conf)
                        state['ocr_crops'].append(plate_crop.copy())
                        state['last_ocr_frame'] = frame_idx
                        state['ocr_attempts'].append(frame_idx)
                        print(f"OCR attempt {len(state['ocr_texts'])} for ID {tid} at frame {frame_idx}, text: '{final_text}'")

                        if len(state['ocr_texts']) == OCR_ATTEMPTS_NEEDED:
                            finalize_ocr_for_track(tid, state, x1, y1, x2, y2, conf, frame_idx, loop)

                    if not in_roi and was_in_roi and len(state['ocr_texts']) > 0 and not state['ocr_done']:
                        finalize_ocr_for_track(tid, state, x1, y1, x2, y2, conf, frame_idx, loop)

        with roi_lock:
            cur_roi = roi_box.copy()
        cv2.rectangle(annotated, (cur_roi[0], cur_roi[1]), (cur_roi[2], cur_roi[3]), (255,0,0), 2)

        with frame_lock:
            try:
                latest_frame_bytes = encode_jpg(annotated)
            except Exception:
                latest_frame_bytes = encode_jpg(proc_frame)

        target_interval = 1.0 / max(1.0, float(FPS))
        loop_end_ts = time.perf_counter()
        elapsed = loop_end_ts - (start_read_ts if 'start_read_ts' in locals() else (time.perf_counter() - target_interval))
        sleep_time = target_interval - elapsed
        if sleep_time > 0:
            time.sleep(min(sleep_time, target_interval))
        else:
            time.sleep(0.0005)

async def broadcaster_task():
    print("Broadcaster started.")
    while True:
        det = await detection_queue.get()
        if det is None: continue
        message = json.dumps(det)
        async with websockets_lock:
            ws_list = list(connected_websockets)
        to_remove = []
        for ws in ws_list:
            try:
                await ws.send_text(message)
            except Exception:
                to_remove.append(ws)
        if to_remove:
            async with websockets_lock:
                for w in to_remove:
                    if w in connected_websockets:
                        connected_websockets.remove(w)

def mjpeg_generator():
    boundary = b'--frame\r\n'
    last_send = time.perf_counter()
    while True:
        with frame_lock:
            frame = latest_frame_bytes
        if frame is None:
            black = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            frame = encode_jpg(black)
        yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        interval = 1.0 / max(1.0, float(FPS))
        now = time.perf_counter()
        elapsed = now - last_send
        to_sleep = interval - elapsed
        last_send = now
        if to_sleep > 0:
            time.sleep(to_sleep)
        else:
            time.sleep(0.0005)

@router.get("/video")
def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/frame_info")
async def frame_info():
    return {"frame_w": FRAME_W, "frame_h": FRAME_H, "fps": FPS}

@router.post("/set_roi")
async def set_roi(box: ROIModel):
    with roi_lock:
        roi_box[0] = max(0, min(FRAME_W-1, box.x1))
        roi_box[1] = max(0, min(FRAME_H-1, box.y1))
        roi_box[2] = max(0, min(FRAME_W-1, box.x2))
        roi_box[3] = max(0, min(FRAME_H-1, box.y2))
    return JSONResponse({"success": True, "roi": roi_box.copy()})

@router.post("/start_video")
async def start_video():
    global video_paused
    with video_control_lock:
        video_paused = False
    return JSONResponse({"success": True})

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    async with websockets_lock:
        connected_websockets.append(ws)
    try:
        while True:
            msg = await ws.receive_text()
            try:
                obj = json.loads(msg)
                if obj.get("cmd") == "ping":
                    await ws.send_text(json.dumps({"pong": True}))
            except Exception:
                pass
    except (WebSocketDisconnect, Exception):
        async with websockets_lock:
            if ws in connected_websockets:
                connected_websockets.remove(ws)

@router.post("/upload_video")
async def upload_video_api(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File not video")
        name = f"uploaded_{uuid.uuid4().hex}_{file.filename}"
        path = os.path.join(OUTPUT_DIR, name)
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
        with video_file_lock:
            global current_video_path
            current_video_path = path
        return JSONResponse({"success": True, "path": path})
    except Exception as e:
        print("upload_video error:", e)
        raise HTTPException(status_code=500, detail=str(e))