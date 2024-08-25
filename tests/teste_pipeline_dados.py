
import unittest
from src.pipeline_dados import transformar_dados, extrair_dados
import pandas as pd
from unittest.mock import patch

class TestPipelineDados(unittest.TestCase):
    def test_transformar_dados(self):
        df = pd.DataFrame({
            'valor_emprestimo': [1000, 2000],
            'renda_anual': [50000, 80000],
        })
        df_transformado = transformar_dados(df)
        self.assertIn('emprestimo_para_renda', df_transformado.columns)

    @patch('src.pipeline_dados.os.getenv')
    def test_extrair_dados(self, mock_getenv):
        mock_getenv.return_value = 'C:/Users/Marcel/Desktop/Marcel/financial-default-prediction/dados/brutos/dados_clientes.csv'
        df = extrair_dados()
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == "__main__":
    unittest.main()
