from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
    
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Colunas no formato dtype são substituídas pelo valor mais frequente na coluna

        Colunas de outros tipos são substituídas com a média dos valores 
        da coluna

        O objetivo é evitar erros envolvendo diferentes formatos na hora de substituir os valores

        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
