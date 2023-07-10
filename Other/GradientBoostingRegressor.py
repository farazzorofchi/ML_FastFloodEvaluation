import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.externals import joblib


data = pd.read_csv('FEMA_Data_Cleaned_Regression.csv')
data.drop('TIV', axis=1, inplace=True)
data.drop('LOSS', axis=1, inplace=True)

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

all_model_GB = Pipeline([
    ('ohe', OHE_final),
    ('Predictor_gb', ensemble.GradientBoostingRegressor(loss='ls'))
])
parameters = {
    'Predictor_gb__learning_rate': [0.1],
    'Predictor_gb__max_depth': [13, 14, 15],
    'Predictor_gb__min_samples_split': [4],
    'Predictor_gb__criterion': ['mse']
}
str_model_GB = GridSearchCV(all_model_GB, parameters, cv=10, verbose=2)
str_model_GB.fit(X_train_str, y_train_str)

print('best score of test data:', str_model_GB.score(X_test_str, y_test_str))
print('best parameters of gridsearchcv:', str_model_GB.best_params_)
joblib.dump(str_model_GB.best_estimator_, 'str_model_GB_best_params.pkl')
