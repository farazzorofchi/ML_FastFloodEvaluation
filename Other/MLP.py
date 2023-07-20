import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib


data = pd.read_csv('FEMA_Data_Cleaned_Regression.csv')
X = data.drop(['total_loss_ratio', 'loss_ratio_building', 'loss_ratio_content'], axis=1)
y_all = data['total_loss_ratio']
y_building = data['loss_ratio_building']
y_content = data['loss_ratio_content']

transformer_name = 'OHE_on_all_categorical_features'
transformer = OneHotEncoder(handle_unknown='ignore')
columns_to_encode = [
    'agriculture_structure_indicator',
    'basement_enclosure_crawlspace_type',
    'condominium_coverage_type_code',
    'elevated_building_indicator',
    'rated_flood_zone',
    'house_worship',
    'location_of_contents',
    'number_of_floors_in_the_insured_building',
    'non_profit_indicator',
    'obstruction_type',
    'occupancy_type',
    'post_f_i_r_m_construction_indicator',
    'reported_zip_code'
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
