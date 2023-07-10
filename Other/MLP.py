import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
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

mlp_model = Pipeline([
    ('ohe', OHE_final),
    ('mlp', MLPRegressor())
])
parameters = {
    'mlp__hidden_layer_sizes': [
        (50, 80, 70, 60, 50, 40, 30, 20, 10,),
        (50, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10,),
        (50, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 35, 40, 45, 50,)
    ],
    'mlp__alpha': [0.001, 0.0001]
}
cnt_model = GridSearchCV(mlp_model, parameters, cv=10, verbose=2, n_jobs=1)
cnt_model.fit(X_train_cnt, y_train_cnt)

print('best score of test data:', cnt_model.score(X_test_cnt, y_test_cnt))
print('best parameters of gridsearchcv:', cnt_model.best_params_)
joblib.dump(cnt_model.best_estimator_, 'cnt_model_MLP_best_params.pkl')
