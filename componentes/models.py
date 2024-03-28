from pydantic import BaseModel

class Task(BaseModel):
    id: int
    task: str
    completed: bool


class Prediction_Input(BaseModel):
    id: int
    Mean_Integrated: float
    SD: float
    EK: float
    Skewness: float
    Mean_DMSNR_Curve: float
    SD_DMSNR_Curve: float
    EK_DMSNR_Curve: float
    Skewness_DMSNR_Curve: float

class Prediction_Output(BaseModel):
    id: int
    Mean_Integrated: float
    SD: float
    EK: float
    Skewness: float
    Mean_DMSNR_Curve: float
    SD_DMSNR_Curve: float
    EK_DMSNR_Curve: float
    Skewness_DMSNR_Curve: float
    Class: int