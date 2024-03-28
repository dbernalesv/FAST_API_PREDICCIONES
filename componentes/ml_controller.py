from fastapi import APIRouter, HTTPException, status
from models import Prediction_Input
from models import Prediction_Output
from pickle import load
import pandas as pd
import xgboost as xgb


router = APIRouter()


@router.get('/ml',status_code=status.HTTP_201_CREATED, response_model=Prediction_Output)
def get_preds(p_id:int,Mean_Integrated:float,SD:float,EK:float,Skewness:float,Mean_DMSNR_Curve:float,SD_DMSNR_Curve: float,EK_DMSNR_Curve: float,
               Skewness_DMSNR_Curve: float):
    
    numerical_columns = ['Mean_Integrated','SD','EK','Skewness','Mean_DMSNR_Curve','SD_DMSNR_Curve','EK_DMSNR_Curve','Skewness_DMSNR_Curve']

    # importar model
    xgb_model = load(open('tuned_xgb_model.pkl', 'rb'))
    # importar scaler
    scaler = load(open('transformer.pkl', 'rb'))

    # Datos de entrada 
    data = {'Mean_Integrated': Mean_Integrated,
            'SD': SD,
            'EK': EK,
            'Skewness': Skewness,
            'Mean_DMSNR_Curve': Mean_DMSNR_Curve,
            'SD_DMSNR_Curve': SD_DMSNR_Curve,
            'EK_DMSNR_Curve': EK_DMSNR_Curve,
            'Skewness_DMSNR_Curve':Skewness_DMSNR_Curve
             }
    
    features = pd.DataFrame(data, index=[0])

    # Convertir los datos de entrada
    new_data = scaler.transform(features)
    new_data_stand = pd.DataFrame(new_data,columns=numerical_columns)

    # Realizar la predicción utilizando el modelo 
    prediction = xgb_model.predict(new_data_stand)
    prediction_dict = Prediction_Output(id=p_id,
                                        Mean_Integrated=Mean_Integrated,
                                        SD=SD,
                                        EK= EK,
                                        Skewness=Skewness,
                                        Mean_DMSNR_Curve=Mean_DMSNR_Curve,
                                        SD_DMSNR_Curve=SD_DMSNR_Curve,
                                        EK_DMSNR_Curve=EK_DMSNR_Curve,
                                        Skewness_DMSNR_Curve=Skewness_DMSNR_Curve,
                                        Class=int(prediction))

    return prediction_dict



