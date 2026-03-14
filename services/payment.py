from typing import Dict, Any
import re

registered_cars_db = [
    {"licensePlate": "41A63723", "amount": 1000000, "date": "20/07/2024"},
    {"licensePlate": "51B12345", "amount": 500000, "date": "20/07/2024"}
]

def norm_plate(s: str) -> str:
    if not s:
        return ""
    s2 = re.sub(r'[^0-9A-Za-z]', '', s)
    return s2.upper()

def deduct_amount(plate: str, cost: int = 20000) -> Dict[str, Any]:
    nplate = norm_plate(plate)
    for car in registered_cars_db:
        if norm_plate(car.get("licensePlate", "")) == nplate:
            try:
                amount = int(car.get("amount", 0))
            except Exception:
                amount = 0
            if amount >= cost:
                car["amount"] = amount - cost
                return {"status": "charged", "remaining": car["amount"], "deducted": cost}
            else:
                car["amount"] = max(0, amount)
                return {"status": "no_balance", "remaining": car["amount"], "deducted": 0}
    return {"status": "unregistered"}

def get_registered_cars():
    return registered_cars_db

def register_car(license_plate: str, amount: int):
    existing = next((c for c in registered_cars_db if norm_plate(c.get("licensePlate","")) == norm_plate(license_plate)), None)
    if existing:
        existing["amount"] = int(amount)
    else:
        registered_cars_db.append({
            "licensePlate": license_plate,
            "amount": int(amount),
            "date": __import__('time').strftime("%d/%m/%Y")
        })

def delete_car(license_plate: str):
    global registered_cars_db
    car_to_delete = next((c for c in registered_cars_db if norm_plate(c.get("licensePlate","")) == norm_plate(license_plate)), None)
    if car_to_delete:
        registered_cars_db.remove(car_to_delete)
        return True
    return False