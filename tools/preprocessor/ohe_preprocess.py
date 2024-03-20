import numpy as np
import polars as pl 
from sklearn.preprocessing import OneHotEncoder


class Preprocessor():

    def __init__(self, params):
        self.pk_cols = ['PassengerId']
        self.num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
        self.cat_cols = ['Pclass', 'Sex', 'Embarked']

        self.feature_cols = ['Age', 'SibSp', 'Parch', 'Fare']
        self.label_cols = ['Survived']




    def _make_one_hot_encoder(self, df: pl.DataFrame, cols: str) -> OneHotEncoder:
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df[cols])
        self.one_hot_encoder_cols = []
        for i, col in enumerate(cols):
            for name in self.one_hot_encoder.categories_[i]:
                self.one_hot_encoder_cols.append(f'{col}_{name}')
                self.feature_cols.append(f'{col}_{name}')
        return self.one_hot_encoder
        

    def _one_hot_encoding(self, df: pl.DataFrame, cols: str) -> pl.DataFrame:
        encoded_data = self.one_hot_encoder.transform(df[cols])
        encoded_df = pl.DataFrame(encoded_data, schema=self.one_hot_encoder_cols)
        return encoded_df
            






    def get_labels(self, train_df: pl.DataFrame) -> pl.DataFrame:
        label_df = train_df[self.pk_cols + self.label_cols]
        return label_df

    def fit(self, train_df: pl.DataFrame) -> pl.DataFrame:
        self._make_one_hot_encoder(train_df, self.cat_cols)



    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        num_df = df[self.pk_cols + self.num_cols].fill_nan(0).fill_null(0)
        cat_df = self._one_hot_encoding(df, self.cat_cols)
        for pk_col in self.pk_cols:
            cat_df = cat_df.with_columns(
                df[pk_col]
            )
        encoded_df = num_df.join(cat_df, how='left', on=self.pk_cols)
        return encoded_df

    def transform_train_df(self, train_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_train_df = self.transform(train_df)
        y_train_df = self.get_labels(train_df)
        train_df = X_train_df.join(y_train_df, how='left', on=self.pk_cols)
        X = train_df[self.feature_cols].to_numpy()
        y = train_df[self.label_cols].to_numpy().reshape(-1)
        return X, y
    
    def transform_test_df(self, test_df: pl.DataFrame) -> np.ndarray:
        test_df = self.transform(test_df)
        X = test_df[self.feature_cols].to_numpy()
        return X


if __name__ == '__main__':

    train_df = pl.read_csv('./data/train.csv')
    test_df = pl.read_csv('./data/test.csv')
    preprocessor = Preprocessor()
    preprocessor.fit(train_df)
    print(preprocessor.transform(train_df))
    print(preprocessor.transform(test_df))