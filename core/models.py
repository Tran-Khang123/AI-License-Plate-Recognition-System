from pydantic import BaseModel

class ROIModel(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class RegisterCarData(BaseModel):
    licensePlate: str
    amount: int

class DeleteCarData(BaseModel):
    licensePlate: str