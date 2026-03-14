import os
import time
import base64
import uuid
import math
import warnings
import cv2
import numpy as np
from ultralytics import YOLO
from config import OUTPUT_DIR, OCR_ATTEMPTS_NEEDED, OCR_FRAME_INTERVAL, MIN_VOTE_LENGTH, FAILED_OCR_DIR
from services.payment import deduct_amount
from datetime import datetime

warnings.filterwarnings('ignore')

def process_license_plate(image_path, conf_threshold=0.5, ocr=None, char_model=None, plate_model=None):
    try:
        start_time = time.time()
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Không thể đọc ảnh đầu vào")

        results = plate_model(image) if plate_model is not None else []
        all_plates = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy().astype(int))
                        confidence = float(box.conf[0].cpu().numpy())
                    except Exception:
                        continue
                    if confidence >= conf_threshold:
                        all_plates.append({'bbox':[x1,y1,x2,y2], 'confidence': float(confidence)})

        if len(all_plates) == 0:
            return {'error': f'Không detect được biển số nào với ngưỡng >= {conf_threshold}'}

        all_plates = sorted(all_plates, key=lambda x: x['confidence'], reverse=True)
        detected_plates = []

        for plate_info in all_plates:
            x1, y1, x2, y2 = plate_info['bbox']
            shrink = 4
            x1s = max(0, x1 + shrink)
            y1s = max(0, y1 + shrink)
            x2s = min(image.shape[1] - 1, x2 - shrink)
            y2s = min(image.shape[0] - 1, y2 - shrink)
            plate_region = image[y1s:y2s, x1s:x2s]
            crop_filename = f"plate_crop_{uuid.uuid4().hex}.jpg"
            crop_path = os.path.join(OUTPUT_DIR, crop_filename)
            cv2.imwrite(crop_path, plate_region)

            rec_polys = []
            try:
                if ocr is not None:
                    ocr_result = ocr.predict(input=crop_path)
                    if ocr_result and isinstance(ocr_result, list):

                        for i, res in enumerate(ocr_result):
                            json_filename = f"paddle_result_{uuid.uuid4().hex}_{i}.json"
                            json_path = os.path.join(OUTPUT_DIR, json_filename)
                            res.save_to_json(json_path)
                            import json
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            rec_polys.extend(data.get('rec_polys', []))
                            if os.path.exists(json_path):
                                os.remove(json_path)
                else:
                    h, w = plate_region.shape[:2]
                    rec_polys = [[[0,0],[w-1,0],[w-1,h-1],[0,h-1]]]
            except Exception as e:
                print("OCR error (paddle):", e)
                h, w = plate_region.shape[:2]
                rec_polys = [[[0,0],[w-1,0],[w-1,h-1],[0,h-1]]]

            all_angles = []
            for poly in rec_polys:
                poly_array = np.array(poly)
                if len(poly_array) >= 2:
                    top_edge = poly_array[1] - poly_array[0]
                    angle_rad = math.atan2(top_edge[1], top_edge[0])
                    angle_deg = math.degrees(angle_rad)
                    all_angles.append(angle_deg)

            mean_angle = float(np.mean(all_angles)) if all_angles else 0.0
            h, w = plate_region.shape[:2]
            center = (w // 2, h // 2)
            rotated_plate = plate_region
            if abs(mean_angle) > 0.5:
                M = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
                rotated_plate = cv2.warpAffine(plate_region, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            corrected_plate_image = rotated_plate
            final_text = "N/A"
            characters = []

            try:
                if char_model is not None:
                    results_char = char_model(corrected_plate_image)
                    for result in results_char:
                        boxes = getattr(result, "boxes", None)
                        if boxes is None: continue
                        for box in boxes:
                            try:
                                xx1, yy1, xx2, yy2 = map(int, box.xyxy[0].cpu().numpy().astype(int))
                                cconf = float(box.conf[0].cpu().numpy())
                                cls = int(box.cls[0].cpu().numpy())
                                name = char_model.names[cls] if hasattr(char_model, 'names') else str(cls)
                                cx = (xx1 + xx2) / 2.0
                                cy = (yy1 + yy2) / 2.0
                                characters.append({'bbox':[xx1,yy1,xx2,yy2],'confidence':cconf,'class_name':str(name),'cx':cx,'cy':cy,'w':xx2-xx1,'h':yy2-yy1})
                            except Exception:
                                continue
            except Exception as e:
                print("char_model error (image):", e)

            characters = [c for c in characters if c['confidence'] >= 0.6]
            if characters:
                heights = [c['h'] for c in characters if c['h'] > 0]
                median_h = float(np.median(heights)) if heights else 20.0
                row_thresh = max(8.0, median_h * 0.6)
                chars_by_cy = sorted(characters, key=lambda x: x['cy'])
                rows = []
                for c in chars_by_cy:
                    placed = False
                    for r in rows:
                        if abs(c['cy'] - r['mean_y']) <= row_thresh:
                            r['items'].append(c)
                            r['mean_y'] = sum(it['cy'] for it in r['items']) / len(r['items'])
                            placed = True
                            break
                    if not placed:
                        rows.append({'mean_y': c['cy'], 'items': [c]})

                rows = sorted(rows, key=lambda r: r['mean_y'])
                texts_per_row = []
                for r in rows:
                    row_items = sorted(r['items'], key=lambda x: x['cx'])
                    row_text = ''.join([it['class_name'] for it in row_items])
                    texts_per_row.append(row_text)

                final_text = ''.join(texts_per_row) if texts_per_row else "N/A"

            try:
                _, buffer = cv2.imencode('.jpg', corrected_plate_image)
                img_str = base64.b64encode(buffer).decode()
            except Exception:
                img_str = ""

            payment = deduct_amount(final_text)
            current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            detected_plates.append({
                'license_plate': final_text,
                'result_image': img_str,
                'payment': payment,
                'timestamp': current_time,
            })

            if os.path.exists(crop_path):
                try: os.remove(crop_path)
                except: pass

        processing_time = time.time() - start_time
        return {
            'success': True,
            'detected_plates': detected_plates,
            'total_plates': len(detected_plates),
            'processing_time': f'{processing_time:.2f} giây'
        }
    except Exception as e:
        print("Lỗi trong process_license_plate:", e)
        return {'error': f'Lỗi xử lý: {str(e)}'}