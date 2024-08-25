import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def treinar_modelo(df):
    """Treina o modelo de machine learning."""
    X = df.drop('inadimplencia', axis=1)
    y = df['inadimplencia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    joblib.dump(modelo, 'C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/modelos/modelo_inadimplencia.pkl')
    return modelo

if __name__ == "__main__":
    df = pd.read_csv('C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/dados/features/features_clientes.csv')
    df['inadimplencia'] = pd.read_csv('C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/dados/processados/dados_clientes_transformados.csv')['inadimplencia']
    modelo = treinar_modelo(df)
