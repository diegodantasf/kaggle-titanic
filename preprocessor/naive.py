import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

class NaiveProcessor:
    def __init__(self, train_df):
        assert (type(train_df) == type(pd.DataFrame()))
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        X, y = self.clean_transform(train_df)
        self.scaler_X = self.scaler_X.fit(X)
        self.scaler_y = self.scaler_y.fit(y)

    def __call__(self, df, with_label=False, shuffle=False):
        # Cleaning
        X, y = self.clean_transform(df, with_label)

        # Shuffle
        if shuffle:
            np.random.seed(1337) # Fixed seed
            permutation = np.arange(X.shape[0])
            np.random.shuffle(permutation)
            X = X[permutation]; 
            if with_label:
                y = y[permutation]

        # Normalize
        X = self.scaler_X.transform(X)
        if with_label:
            y = self.scaler_y.transform(y)
        
        return X, y
    
    def clean_transform(self, df, with_label=True):
        df = df.copy()

        drop_names = ['Name', 'Ticket', 'PassengerId']
        to_categorical = ['Embarked', 'Cabin', 'Sex']

        df = df.drop(columns=drop_names)
        for s in to_categorical:
            df[s] = df[s].astype('category').cat.codes

        X = df.loc[:, 'Pclass':].to_numpy()
        y = None
        if with_label:
            y = df.loc[:, 'Survived'].to_numpy().reshape((-1, 1))

        X = np.nan_to_num(X, nan=0)

        return X, y
       
    def split(self, X, y=None, percentage=0.75, with_label=False):
        assert (type(X) == type(np.array()))
        idx = int(X.shape[0] * percentage)
        X1, X2 = X[:idx], X[idx:]
        if with_label:
            y1, y2 = y[:idx], y[idx:]
            return X1, y1, X2, y2
        return X1, X2