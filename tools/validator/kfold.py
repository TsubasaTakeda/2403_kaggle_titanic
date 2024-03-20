from sklearn.model_selection import KFold


class Validator():

    def __init__(self, params):
        self.params = params
        self.kf = KFold(**self.params)
        
    def validate(self, train_df, predict_func):

        results = []

        for train_index, test_index in self.kf.split(train_df):
            kf_train_df = train_df[train_index]
            kf_test_df = train_df[test_index]

            df_result_df = predict_func(kf_train_df, kf_test_df)

            results.append(df_result_df)
        
        return results