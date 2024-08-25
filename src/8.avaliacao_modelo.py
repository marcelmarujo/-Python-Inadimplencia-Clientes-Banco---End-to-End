import pandas as pd
from sklearn.metrics import classification_report
import joblib

def avaliar_modelo(df_features, df_target):
    """Avalia o modelo usando dados de teste."""
    modelo = joblib.load('C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/modelos/modelo_inadimplencia.pkl')
    y_pred = modelo.predict(df_features)
    return classification_report(df_target, y_pred)

if __name__ == "__main__":
    df_features = pd.read_csv('C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/dados/features/features_clientes.csv')
    df_target = pd.read_csv('C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/dados/processados/dados_clientes_transformados.csv')['inadimplencia']
    relatorio = avaliar_modelo(df_features, df_target)
    print(relatorio)
