import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

load_dotenv()

def criar_features(df):
    """Cria novas features e escala os dados."""
    df['divida_para_renda'] = df['divida'] / df['renda_anual']
    features = df[['emprestimo_para_renda', 'divida_para_renda', 'pontuacao_credito']]
    scaler = StandardScaler()
    features_escaladas = scaler.fit_transform(features)
    return pd.DataFrame(features_escaladas, columns=features.columns)

if __name__ == "__main__":
    df = pd.read_csv('C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/dados/processados/dados_clientes_transformados.csv')
    df_features = criar_features(df)
    df_features.to_csv('C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/dados/features/features_clientes.csv', index=False)
