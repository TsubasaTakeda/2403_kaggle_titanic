import polars as pl
from importlib import import_module


class Predictor():

    def __init__(self, params):
        self.params = params
        self.Preprocessor = import_module(params['pre_params']['lib_name']).Preprocessor
        self.Model = import_module(params['model_params']['lib_name']).Model 
        self.Postprocessor = import_module(params['post_params']['lib_name']).Postprocessor 

        self.pre_params = params['pre_params']['params']
        self.model_params = params['model_params']['params']
        self.post_params = params['post_params']['params']

    def predict(self, train_df: pl.DataFrame, test_df: pl.DataFrame):
        
        # 前処理
        pre = self.Preprocessor(self.pre_params)
        pre.fit(train_df)
        X_train, y_train = pre.transform_train_df(train_df)
        X_test = pre.transform_test_df(test_df)

        # モデルの作成と予測
        model = self.Model(self.model_params)
        model.train(X_train, y_train)
        pred_y = model.predict(X_test)

        # 後処理
        post = self.Postprocessor(self.post_params)
        submit_y = post.transform(pred_y)

        # DataFrame変換
        result_df = test_df[pre.pk_cols].with_columns(
            Survived_before_postprocess = pred_y, 
            Survived = submit_y
        )

        return result_df