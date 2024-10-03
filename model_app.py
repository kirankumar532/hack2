from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

class Input(BaseModel):
    CONSOLE:object 
    YEAR:int
    CATEGORY:object 
    PUBLISHER:object 
    RATING:object 
    CRITICS_POINTS:float
    USER_POINTS:float


class Output(BaseModel):
    SalesInMillions:float


@app.post("/predict")
def predict(data: Input) -> Output:
    #Input
    X_input = pd.DataFrame([{'CONSOLE':data.CONSOLE,'YEAR':data.YEAR,'CATEGORY':data.CATEGORY,'PUBLISHER':data.PUBLISHER,'RATING':data.RATING,'CRITICS_POINTS':data.CRITICS_POINTS,'USER_POINTS':data.USER_POINTS}])
    
    #load the model
    model = joblib.load('gaming_data_pred.pkl')
    
    #predict using the model
    prediction = model.predict(X_input)
    
    #output
    return Output(SalesInMillions = prediction)


