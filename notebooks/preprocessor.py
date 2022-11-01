
import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessor : 
    def __init__ (self, df):
        self.df = df.copy()
        
    def handle_missing_values (self) :
        self.df.fillna(0, inplace=True)

    def select_features(self):
        self.df = self.df.iloc[:,2:]

    def standard_scaler(self):
        s = StandardScaler()
        self.df = s.fit_transform(self.df)
      
    def transform (self) : 
        self.handle_missing_values()
        self.select_features()
        self.standard_scaler()
        return self.df
