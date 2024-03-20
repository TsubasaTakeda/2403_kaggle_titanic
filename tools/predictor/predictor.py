import polars as pl
from importlib import import_module

from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from dataclasses import dataclass 


@dataclass(frozen=True)
class ValidationResult():

    fold_result_df: list[pl.DataFrame]
    whole_result_df: pl.DataFrame
    each_fold_score: list[dict[str,float]]
    whole_score: dict[str,float]






class Predictor():

    def __init__(self, params):
        self.params = params
        self.Preprocessor = import_module(params['pre_params']['lib_name']).Preprocessor
        self.Model = import_module(params['model_params']['lib_name']).Model 
        self.Postprocessor = import_module(params['post_params']['lib_name']).Postprocessor 
        self.Validator = import_module(params['valid_params']['lib_name']).Validator

        self.pre_params = params['pre_params']['params']
        self.model_params = params['model_params']['params']
        self.post_params = params['post_params']['params']
        self.valid_params = params['valid_params']['params']



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

        # DataFrame変換 (コンペに合わせて変える必要あり)
        result_df = test_df[pre.pk_cols].with_columns(
            pred_before_postprocessor = pred_y, 
            pred = submit_y
        )

        return result_df
    


    def validate(self, train_df: pl.DataFrame):
        """コンペに合わせて、join方法等を変える必要あり"""
        
        validator = self.Validator(self.valid_params)
        fold_result_df = validator.validate(train_df, self.predict)
        for i in range(len(fold_result_df)):
            fold_result_df[i] = fold_result_df[i].join(train_df[['PassengerId', 'Survived']], how='left', on=['PassengerId'])
        whole_result_df = pl.concat(fold_result_df, how='vertical').sort(pl.col('PassengerId'))

        each_fold_score = {}
        whole_score = {}

        # accuracy
        each_fold_score['acc'] = [self._calc_acc(df) for df in fold_result_df]
        whole_score['acc'] = self._calc_acc(whole_result_df)

        # f1 macro
        each_fold_score['f1_macro'] = [self._calc_f1_macro(df) for df in fold_result_df]
        whole_score['f1_macro'] = self._calc_f1_macro(whole_result_df)

        # log loss
        each_fold_score['log_loss'] = [self._calc_log_loss(df) for df in fold_result_df]
        whole_score['log_loss'] = self._calc_log_loss(whole_result_df)

        # ROC AUC
        each_fold_score['roc_auc'] = [self._calc_roc_auc(df) for df in fold_result_df]
        whole_score['roc_auc'] = self._calc_roc_auc(whole_result_df)

        result = ValidationResult(fold_result_df, whole_result_df, each_fold_score, whole_score)
        return result


    def _calc_acc(self, result_df: pl.DataFrame):
        y_true = result_df['Survived'].to_numpy().reshape(-1)
        y_pred = result_df['pred'].to_numpy().reshape(-1)
        acc = accuracy_score(y_true, y_pred)
        return acc
    
    def _calc_f1_macro(self, result_df: pl.DataFrame):
        y_true = result_df['Survived'].to_numpy().reshape(-1)
        y_pred = result_df['pred'].to_numpy().reshape(-1)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        return f1_macro

    def _calc_log_loss(self, result_df: pl.DataFrame):
        y_true = result_df['Survived'].to_numpy().reshape(-1)
        y_pred = result_df['pred_before_postprocessor'].to_numpy().reshape(-1)
        lloss = log_loss(y_true, y_pred)
        return lloss
    
    def _calc_roc_auc(self, result_df: pl.DataFrame):
        y_true = result_df['Survived'].to_numpy().reshape(-1)
        y_pred = result_df['pred_before_postprocessor'].to_numpy().reshape(-1)
        auc = roc_auc_score(y_true, y_pred)
        return auc
