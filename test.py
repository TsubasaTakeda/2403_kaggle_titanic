import polars as pl
from tools.predictor.predictor import Predictor


pre_params = {
    'lib_name': 'tools.preprocessor.ohe_preprocess', 
    'params': {}
}

model_params = {
    'lib_name': 'tools.model.lr_logistic_model',
    'params': {
        'random_state': 42, 
        'max_iter': 1000
    }
}

post_params = {
    'lib_name': 'tools.postprocessor.threshold',
    'params': {
        'threshold': 0.5
    }
}

valid_params = {
    'lib_name': 'tools.validator.kfold', 
    'params': {
        'n_splits': 5, 
        'shuffle': True, 
        'random_state': 42
    }
}

















params = {
    'pre_params': pre_params, 
    'model_params': model_params, 
    'post_params': post_params, 
    'valid_params': valid_params
}
train_df = pl.read_csv('./data/train.csv')
test_df = pl.read_csv('./data/test.csv')


predictor = Predictor(params)
# result = predictor.predict(train_df, test_df)
result = predictor.validate(train_df)

print(result)