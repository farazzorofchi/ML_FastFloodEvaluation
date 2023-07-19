import random

import pandas as pd
import numpy as np


data = pd.read_csv('openFEMA_claims20190831.csv')
data['amount_paid_on_increased_cost_of_compliance_claim'] = (
    data['amount_paid_on_increased_cost_of_compliance_claim'].fillna(0)
)
data['amount_paid_on_building_claim'] = data['amount_paid_on_building_claim'].fillna(0)
data['amount_paid_on_contents_claim'] = data['amount_paid_on_contents_claim'].fillna(0)
data['total_insurance_value'] = (
    data['total_building_insurance_coverage'] +
    data['total_contents_insurance_coverage']
)
data['total_loss'] = (
    data['amount_paid_on_building_claim'] +
    data['amount_paid_on_contents_claim']
)
data['loss_ratio_building'] = (
    data['amount_paid_on_building_claim'] /
    data['total_building_insurance_coverage']
)
data['loss_ratio_content'] = (
    data['amount_paid_on_contents_claim'] /
    data['total_contents_insurance_coverage']
)
data['total_loss_ratio'] = data['total_loss'] / data['total_insurance_value']

# Keep only year for year built of the building
original_construction_date = []
for i in range(0, len(data)):
    try:
        original_construction_date.append(int(data['original_construction_date'][i][:4]))
    except:
        original_construction_date.append(random.randint(1970, 2019))
data['original_construction_date'] = original_construction_date

# Generalize Flood Zones:
rated_flood_zone = []
for i in range(0, len(data)):
    try:
        if data['rated_flood_zone'][i][:2] == 'AO':
            rated_flood_zone.append('AO')
        elif data['rated_flood_zone'][i][:2] == 'AH':
            rated_flood_zone.append('AH')
        elif data['rated_flood_zone'][i][:2] == 'AR':
            rated_flood_zone.append('AR')
        elif data['rated_flood_zone'][i][:2] == 'AE':
            rated_flood_zone.append('AE')
        elif data['rated_flood_zone'][i][:2] == 'VE':
            rated_flood_zone.append('VE')
        elif data['rated_flood_zone'][i][:3] == 'A99':
            rated_flood_zone.append('A99')
        elif data['rated_flood_zone'][i][0] == 'A':
            if len(data['rated_flood_zone'][i][0]) > 2:
                rated_flood_zone.append('AE')
            else:
                rated_flood_zone.append('A')
        elif data['rated_flood_zone'][i][0] == 'V':
            if len(data['rated_flood_zone'][i][0]) > 2:
                rated_flood_zone.append('VE')
            else:
                rated_flood_zone.append('V')
        elif data['rated_flood_zone'][i][0] == 'X':
            rated_flood_zone.append('X')
        elif data['rated_flood_zone'][i][0] == 'B':
            rated_flood_zone.append('X')
        elif data['rated_flood_zone'][i][0] == 'C':
            rated_flood_zone.append('X')
        else:
            rated_flood_zone.append('UNK')
    except:
        rated_flood_zone.append('UNK')
data['rated_flood_zone'] = rated_flood_zone

reported_zip_code = []
for i in range(0, len(data)):
    try:
        if int(data['reported_zip_code'][i]) > 100:
            reported_zip_code.append(int(data['reported_zip_code'][i]))
        else:
            reported_zip_code.append(-999999)
    except:
        reported_zip_code.append(-999999)
data['reported_zip_code'] = reported_zip_code

for i in range(0, len(data)):
    if (
        (not np.isnan(data['lowest_floor_elevation'][i])) &
        (not np.isnan(data['base_flood_elevation'][i]))
    ):
        data['elevation_difference'][i] = (
            data['lowest_floor_elevation'][i] -
            data['base_flood_elevation'][i]
        )

data = data[data['elevation_difference'] != 999]
data.reset_index(drop=True, inplace=True)
data.drop('as_of_date', axis=1, inplace=True)
data.drop('county_code', axis=1, inplace=True)
data.drop('census_tract', axis=1, inplace=True)
data.drop('reported_city', axis=1, inplace=True)
data.drop('date_of_loss', axis=1, inplace=True)
data.drop('elevation_certificate_indicator', axis=1, inplace=True)
data.drop('lowest_adjacent_grade', axis=1, inplace=True)
data.drop('lowest_floor_elevation', axis=1, inplace=True)
data.drop('base_flood_elevation', axis=1, inplace=True)
data.drop('original_construction_date', axis=1, inplace=True)
data.drop('original_n_b_date', axis=1, inplace=True)
data.drop('amount_paid_on_increased_cost_of_compliance_claim', axis=1, inplace=True)
data.drop('ratemethod', axis=1, inplace=True)
data.drop('small_business_indicator_building', axis=1, inplace=True)
data.drop('state', axis=1, inplace=True)
data.drop('reported_zip_code', axis=1, inplace=True)
data.drop('primary_residence_indicator', axis=1, inplace=True)
print('len(data) before cleaning =', len(data))

delete_row = data[np.isnan(data['latitude'])].index
data = data.drop(delete_row)
delete_row = data[np.isnan(data['longitude'])].index
data = data.drop(delete_row)
delete_row = data[np.isnan(data['number_of_floors_in_the_insured_building'])].index
data = data.drop(delete_row)

# Keeping only up to 30 feet difference
data = data[
    (data['elevation_difference'] <= 30) &
    (data['elevation_difference'] >= -30)
]
data = data[data['original_construction_date'] >= 1800]
data = data[data['non_profit_indicator'] != '0']
data = data[data['reported_zip_code'] != -999999]
data = data[data['total_insurance_value'] != 0]
data = data[data['obstruction_type'] != '*']
data['obstruction_type'] = pd.to_numeric(data['obstruction_type'])
data['obstruction_type'] = data['obstruction_type'].astype(str)
data['obstruction_type'].fillna('UNK', inplace=True)
data['obstruction_type'][data['obstruction_type'] == 'nan'] = 'UNK'
data['agriculture_structure_indicator'].fillna('N', inplace=True)
data['basement_enclosure_crawlspace_type'].fillna(0, inplace=True)
data['condominium_coverage_type_code'].fillna('N', inplace=True)
data['policy_count'].fillna(1, inplace=True)
data['crs_classification_code'].fillna(0, inplace=True)
data['elevated_building_indicator'].fillna('UNK', inplace=True)
data['rated_flood_zone'].fillna('UNK', inplace=True)
data['house_worship'].fillna('N', inplace=True)
data['location_of_contents'].fillna('UNK', inplace=True)
data['non_profit_indicator'].fillna('N', inplace=True)
data['occupancy_type'].fillna(1.0, inplace=True)
data['post_f_i_r_m_construction_indicator'].fillna('UNK', inplace=True)
data = data[data['rated_flood_zone'] != 'UNK']
data.reset_index(drop=True, inplace=True)
print('len(data) after cleaning =', len(data))

data.to_csv('FEMA_Data_Cleaned_Regression.csv', index=False)
