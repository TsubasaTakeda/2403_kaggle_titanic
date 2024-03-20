import time
import statistics
import mlflow
import polars as pl
from tools.predictor.predictor import Predictor



mlflow.set_tracking_uri(uri='http://localhost:5000')
mlflow.set_experiment('Experiment for KFold Random State')

train_df = pl.read_csv('./data/original_data/train.csv')
test_df = pl.read_csv('./data/original_data/test.csv')




for i in range(43,500):

    print(f'start random_state: {i}')

    now = time.time()

    params = {
        # Libraryの選択
        'pre_lib': 'tools.preprocessor.ohe_preprocess', 
        'model_lib': 'tools.model.lr_logistic_model', 
        'post_lib': 'tools.postprocessor.threshold', 
        'valid_lib': 'tools.validator.kfold', 

        # 前処理のパラメータ

        # モデルのパラメータ
        'model_random_state': 42, 
        'model_max_iter': 1000, 

        # 後処理のパラメータ
        'post_threshold': 0.5, 

        # 検証のパラメータ
        'valid_n_splits': 5, 
        'valid_shuffle': True, 
        'valid_random_state': i
    }









    pre_params = {
        'lib_name': params['pre_lib'], 
        'params': {}
    }
    model_params = {
        'lib_name': params['model_lib'],
        'params': {
            'random_state': params['model_random_state'], 
            'max_iter': params['model_max_iter']
        }
    }
    post_params = {
        'lib_name': params['post_lib'],
        'params': {
            'threshold': params['post_threshold']
        }
    }
    valid_params = {
        'lib_name': params['valid_lib'], 
        'params': {
            'n_splits': params['valid_n_splits'], 
            'shuffle': params['valid_shuffle'], 
            'random_state': params['valid_random_state']
        }
    }
    tmp_params = {
        'pre_params': pre_params, 
        'model_params': model_params, 
        'post_params': post_params, 
        'valid_params': valid_params
    }



    predictor = Predictor(tmp_params)
    result = predictor.validate(train_df)
    submissions = predictor.predict(train_df, test_df)


    submission_df = submissions.select(
        pl.col('PassengerId'), 
        pl.col('pred').alias('Survived')
    )

    filename = f'./data/submissions/{now}.csv'
    submission_df.write_csv(filename)

    kf_filename = f'./data/kf_preds/{now}.csv'
    result.whole_result_df.write_csv(kf_filename)

    for key, value in result.each_fold_score.items():
        result.whole_score[f'{key}_mean'] = statistics.mean(value)
        result.whole_score[f'{key}_variance'] = statistics.variance(value)


    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(result.whole_score)
        mlflow.set_tag('submit_filename', filename)
        mlflow.set_tag('kf_filename', kf_filename)