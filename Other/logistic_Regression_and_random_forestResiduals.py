import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn import base
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

data = pd.read_csv('FEMA_Data_Cleaned_Regression.csv')
data.drop('TIV', axis=1, inplace=True)
data.drop('LOSS', axis=1, inplace=True)

data['loss_ratio_building'][(data['totalbuildinginsurancecoverage'] == 0) & (data['amountpaidonbuildingclaim'] != 0)] = 0
data['loss_ratio_content'][(data['totalcontentsinsurancecoverage'] == 0) & (data['amountpaidoncontentsclaim'] != 0)] = 0

X = data
X = X[X['loss_ratio_overall'] <= 1]
X = X[X['loss_ratio_building'] <= 1]
X = X[X['loss_ratio_content'] <= 1]

y_all = X['loss_ratio_overall']
y_building = X['loss_ratio_building']
y_content = X['loss_ratio_content']


X.drop('loss_ratio_overall', axis=1, inplace=True)
X.drop('loss_ratio_building', axis=1, inplace=True)
X.drop('loss_ratio_content', axis=1, inplace=True)
X.drop('amountpaidonbuildingclaim', axis=1, inplace=True)
X.drop('amountpaidoncontentsclaim', axis=1, inplace=True)
X.drop('totalbuildinginsurancecoverage', axis=1, inplace=True)
X.drop('totalcontentsinsurancecoverage', axis=1, inplace=True)


transformer_name = 'OHE_on_all_categorical_features'
transformer = OneHotEncoder(handle_unknown='ignore')
columns_to_encode = ['agriculturestructureindicator', 'basementenclosurecrawlspacetype', 'condominiumindicator', 'elevatedbuildingindicator', 'floodzone', 'houseworship', 'locationofcontents', 'numberoffloorsintheinsuredbuilding','nonprofitindicator','obstructiontype', 'occupancytype','postfirmconstructionindicator','ZipCode']

OHE_final = ColumnTransformer([
    (transformer_name, transformer, columns_to_encode)],
    remainder='passthrough')


class Predictor_residuals(base.BaseEstimator, base.RegressorMixin):
    def __init__(self, est1, est2):
        self.est1 = est1
        self.est2 = est2

    def fit(self, X, y):
        self.est1.fit(X, y)
        pred = self.est1.predict(X)
        residuals = y - pred
        self.est2.fit(X, residuals)
        return self

    def predict(self, X):
        predictions1 = self.est1.predict(X)
        predictions2 = self.est2.predict(X)
        predictions = predictions1 + predictions2
        return predictions

X_train_str, X_test_str, y_train_str, y_test_str = train_test_split(X, y_building, test_size=0.1, random_state=42)
X_train_cnt, X_test_cnt, y_train_cnt, y_test_cnt = train_test_split(X, y_content, test_size=0.1, random_state=42)

all_model_residual = Pipeline([('ohe', OHE_final),
                            ('Predictor_residuals', Predictor_residuals(Ridge(), RandomForestRegressor()))
                            ])

parameters = {'Predictor_residuals__est1':[Ridge(alpha=0.1),Ridge(alpha=1),Ridge(alpha=10),Ridge(alpha=100),Ridge(alpha=1000)], 'Predictor_residuals__est2':[RandomForestRegressor(max_depth=15),RandomForestRegressor(max_depth=20),RandomForestRegressor(max_depth=22)]}

content_model = GridSearchCV(all_model_residual, parameters, cv = 10, verbose=2, n_jobs=1)
content_model.fit(X_train_str, y_train_str)

print('best score of test data:', content_model.score(X_test_cnt, y_test_cnt))
print('best parameters of gridsearchcv:', content_model.best_params_)

joblib.dump(content_model.best_estimator_, 'cnt_model_residual_best_params.pkl')