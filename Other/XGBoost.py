import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


data = pd.read_csv('FEMA_Data_Cleaned_Regression.csv')
data.drop('TIV', axis=1, inplace=True)
data.drop('LOSS', axis=1, inplace=True)
data['loss_ratio_building'][(data['totalbuildinginsurancecoverage'] == 0) & (
    data['amountpaidonbuildingclaim'] != 0)
] = 0
data['loss_ratio_content'][(data['totalcontentsinsurancecoverage'] == 0) & (
    data['amountpaidoncontentsclaim'] != 0)
] = 0

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
columns_to_encode = [
    'agriculturestructureindicator',
    'basementenclosurecrawlspacetype',
    'condominiumindicator',
    'elevatedbuildingindicator',
    'floodzone',
    'houseworship',
    'locationofcontents',
    'numberoffloorsintheinsuredbuilding',
    'nonprofitindicator',
    'obstructiontype',
    'occupancytype',
    'postfirmconstructionindicator',
    'ZipCode'
]

OHE_final = ColumnTransformer(
    [(transformer_name, transformer, columns_to_encode)],
    remainder='passthrough'
)
X_train_str, X_test_str, y_train_str, y_test_str = train_test_split(
    X, y_building, test_size=0.1, random_state=42
)
X_train_cnt, X_test_cnt, y_train_cnt, y_test_cnt = train_test_split(
    X, y_content, test_size=0.1, random_state=42
)

XGBoost_model = Pipeline([
    ('ohe', OHE_final),
    ('Predictor_xgboost', xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        nthred=3
    ))
])
parameters = {
    'Predictor_xgboost__booster': ['gbtree'],
    'Predictor_xgboost__max_depth': [13, 14, 15],
    'Predictor_xgboost__lambda': [0.01, 0.1]
}
XGBoost_model_grid = GridSearchCV(
    XGBoost_model,
    parameters,
    cv=10,
    verbose=2,
    n_jobs=2
)
XGBoost_model_grid.fit(X_train_str, y_train_str)

print('best score of test data:', XGBoost_model_grid.score(X_test_str, y_test_str))
print('best parameters of gridsearchcv:', XGBoost_model_grid.best_params_)
joblib.dump(
    XGBoost_model_grid.best_estimator_,
    'str_model_XGBoost_best_params.pkl'
)
