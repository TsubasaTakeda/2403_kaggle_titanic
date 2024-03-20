import polars as pl
from tools.preprocess.ohe_preprocess import Preprocessor
from tools.models.lr_logistic_model import Model
from tools.postprocess.threshold import Postprocessor


pre_params = {}
model_params = {
    'random_state': 42, 
    'max_iter': 1000
}
post_params = {
    'threshold': 0.5
}

valid_params = {}


train_df = pl.read_csv('./data/train.csv')
test_df = pl.read_csv('./data/test.csv')

# 前処理
pre = Preprocessor(pre_params)
pre.fit(train_df)

# データの加工
X_train_df = pre.transform(train_df)
test_df = pre.transform(test_df)

y_train_df = pre.get_labels(train_df)
train_df = X_train_df.join(y_train_df, how='left', on=pre.pk_cols)


# モデルの作成と予測
model = Model(model_params)
model.train(train_df[pre.feature_cols].to_numpy(), train_df[pre.label_cols].to_numpy().reshape(-1))
pred_y = model.predict(test_df[pre.feature_cols].to_numpy())


# 後処理
post = Postprocessor(post_params)
submit_y = post.transform(pred_y)


# 提出用ファイルの作成
submit_df = test_df[pre.pk_cols].with_columns(
    Survived = submit_y
)
submit_df.write_csv('./sample_submit.csv')