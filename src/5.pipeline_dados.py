#importando as libs
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def extrair_dados(caminho_arquivo=None):
    """Extrai os dados do arquivo CSV."""
    if caminho_arquivo is None:
        caminho_arquivo = os.getenv('CAMINHO_ARQUIVO_DADOS')
    return pd.read_csv(caminho_arquivo)

def transformar_dados(df):
    """Aplica transformações iniciais aos dados."""
    df['emprestimo_para_renda'] = df['valor_emprestimo'] / df['renda_anual']
    df.dropna(inplace=True)
    return df

def carregar_dados(df, caminho_saida):
    """Carrega os dados transformados para um arquivo CSV."""
    df.to_csv(caminho_saida, index=False)

if __name__ == "__main__":
    df = extrair_dados()
    df_transformado = transformar_dados(df)
    carregar_dados(df_transformado, 'C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/dados/processados/dados_clientes_transformados.csv')
