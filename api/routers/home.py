from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from services.plate_processing import process_license_plate
from services.payment import register_car, get_registered_cars, delete_car
from config import OUTPUT_DIR
import uuid
import os

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    gui_path = os.path.join(os.path.dirname(__file__), "../../templates/GUI.html")
    if not os.path.exists(gui_path):
        return HTMLResponse(content="<h1>GUI.html không tìm thấy!</h1>", status_code=404)
    with open(gui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@router.post("/process_image")
async def process_image_api(image: UploadFile = File(...), confidence_threshold: float = Form(0.5)):
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File không phải là ảnh")
        filename = f"temp_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        contents = await image.read()
        with open(filepath, "wb") as f:
            f.write(contents)

        from main import ocr, char_model, plate_model
        result = process_license_plate(filepath, conf_threshold=confidence_threshold, ocr=ocr, char_model=char_model, plate_model=plate_model)

        if os.path.exists(filepath):
            try: os.remove(filepath)
            except: pass

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        print("Lỗi process_image:", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register_car")
async def register_car_api(data: dict):
    try:
        license_plate = data.get('licensePlate')
        amount = data.get('amount')
        if not license_plate or amount is None:
            raise HTTPException(status_code=400, detail="Thiếu thông tin")
        register_car(license_plate, amount)
        return JSONResponse({
            "success": True,
            "message": f"Đăng ký {license_plate} thành công",
            "updated_cars": get_registered_cars()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get_registered_cars")
async def get_registered_cars_api():
    return JSONResponse({"cars": get_registered_cars()})

@router.delete("/delete_car")
async def delete_car_api(data: dict):
    try:
        license_plate = data.get('licensePlate')
        if not license_plate:
            raise HTTPException(status_code=400, detail="Missing license plate")
        success = delete_car(license_plate)
        if success:
            return JSONResponse({
                "success": True,
                "message": f"Xóa xe {license_plate} thành công",
                "updated_cars": get_registered_cars()
            })
        else:
            raise HTTPException(status_code=404, detail="Car not found")
    except HTTPException:
        raise
    except Exception as e:
        print("delete_car error:", e)
        raise HTTPException(status_code=500, detail=str(e))