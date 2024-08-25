import joblib
import pandas as pd

def prever_inadimplencia(dados):
    """Realiza a previsão de inadimplência para um conjunto de dados."""
    modelo = joblib.load('C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/modelos/modelo_inadimplencia.pkl')
    df = pd.DataFrame([dados])
    previsao = modelo.predict(df)
    return previsao[0]
