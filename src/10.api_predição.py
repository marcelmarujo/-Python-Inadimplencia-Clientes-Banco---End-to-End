from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
modelo = joblib.load('C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/modelos/modelo_inadimplencia.pkl')

@app.post("/prever/")
def prever(dados: dict):
    df = pd.DataFrame([dados])
    previsao = modelo.predict(df)
    return {"probabilidade_inadimplencia": previsao[0]}
