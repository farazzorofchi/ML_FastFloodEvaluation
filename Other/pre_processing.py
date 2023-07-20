import re

import pandas as pd


def to_snake_case(item):
    """Convert input string to 'snake_case' format."""
    regex = r'(?<!^)(?=[A-Z])'
    return re.sub(regex, '_', item).lower()


data_raw = pd.read_csv('FimaNfipClaims.csv', low_memory=False)
data_interim = data_raw.copy()
data_interim.columns = [to_snake_case(item) for item in data_raw.columns]

data_interim = (
    data_interim
    .loc[:, [
        'amount_paid_on_building_claim',
        'amount_paid_on_contents_claim',
        'total_building_insurance_coverage',
        'total_contents_insurance_coverage',

        'agriculture_structure_indicator',
        'basement_enclosure_crawlspace_type',
        'condominium_coverage_type_code',
        'policy_count',
        'crs_classification_code',
        'elevated_building_indicator',
        'elevation_difference',
        'rated_flood_zone',
        'house_worship',
        'location_of_contents',
        'latitude',
        'longitude',
        'number_of_floors_in_the_insured_building',
        'non_profit_indicator',
        'obstruction_type',
        'occupancy_type',
        'post_f_i_r_m_construction_indicator',
        'year_of_loss',
        'original_construction_date',
        'reported_zip_code',
    ]]
    .fillna({
        'amount_paid_on_building_claim': 0.0,
        'amount_paid_on_contents_claim': 0.0,
        'total_building_insurance_coverage': 0.0,
        'total_contents_insurance_coverage': 0.0,

        'agriculture_structure_indicator': 0,  # Not an agriculture structure
        'basement_enclosure_crawlspace_type': 0,  # No basement
        'condominium_coverage_type_code': 'N',  # Not a condominium
        'policy_count': 1,  # Active claim
        'crs_classification_code': 10,  # Classification Credit Percentage 0%.
        'elevated_building_indicator': 0,  # Not an elevated building
        'elevation_difference': 9999.0,  # Default invalid value, will be removed
        'rated_flood_zone': 'UNK',  # Default invalid value, will be removed
        'house_worship': 0,  # Not a house of worship
        'location_of_contents': 1,  # Basement/Enclosure/Crawlspace/Subgrade Crawlspace only
        'latitude': 0.0,
        'longitude': 0.0,
        'number_of_floors_in_the_insured_building': 1,
        'non_profit_indicator': 0,  # Not a non-profit
        'obstruction_type': 91,  # Free of obstruction
        'occupancy_type': 1,  # Single family residence
        'post_f_i_r_m_construction_indicator': 0,  # Not post-FIRM
        'year_of_loss': 1700,  # Default invalid value, will be removed
        'original_construction_date': '1700-01-01T00:00:00.000Z',
        'reported_zip_code': 0,  # Default invalid value, will be removed
    })
    .astype({
        'amount_paid_on_building_claim': 'float64',
        'amount_paid_on_contents_claim': 'float64',
        'total_building_insurance_coverage': 'float64',
        'total_contents_insurance_coverage': 'float64',

        'agriculture_structure_indicator': 'int64',
        'basement_enclosure_crawlspace_type': 'int64',
        'condominium_coverage_type_code': 'str',
        'policy_count': 'int64',
        'crs_classification_code': 'int64',
        'elevated_building_indicator': 'int64',
        'elevation_difference': 'float64',
        'rated_flood_zone': 'str',
        'house_worship': 'int64',
        'location_of_contents': 'int64',
        'latitude': 'float64',
        'longitude': 'float64',
        'number_of_floors_in_the_insured_building': 'int64',
        'non_profit_indicator': 'int64',
        'obstruction_type': 'int64',
        'occupancy_type': 'int64',
        'post_f_i_r_m_construction_indicator': 'int64',
        'year_of_loss': 'int64',
        'original_construction_date': 'str',
        'reported_zip_code': 'int64',
    })
    .assign(
        loss_ratio_building=lambda x: (
            x['amount_paid_on_building_claim'] /
            x['total_building_insurance_coverage']
        ),
        loss_ratio_content=lambda x: (
            x['amount_paid_on_contents_claim'] /
            x['total_contents_insurance_coverage']
        ),
        total_loss_ratio=lambda x: (
            (x['amount_paid_on_building_claim'] + x['amount_paid_on_contents_claim']) /
            (x['total_building_insurance_coverage'] + x['total_contents_insurance_coverage'])
        ),
    )
)

# Keep only year for year built of the building

default_date = '1700-01-01T00:00:00.000Z'
data_interim['original_construction_date'] = (
    data_interim
    .loc[:, 'original_construction_date']
    .replace(
        regex=['1492-10-12T00:00:00.000Z', '0001-01-01T00:00:00.000Z'],
        value=default_date
    )
    .astype('datetime64[ns]')
    .dt.year
)

# Generalize Flood Zones:

data_interim['rated_flood_zone'] = (
    data_interim
    .loc[:, 'rated_flood_zone']
    .replace({
        r'((?!A99)A[0-9]+)': 'AE',
        r'(AH[\w]+)': 'AH',
        r'(AO[\w]+)': 'AO',
        r'(V[0-9]+)': 'VE',
        'B': 'X',
        'C': 'X',
    }, regex=True)
)

data_cleaned = (
    data_interim
    .drop([
        'amount_paid_on_building_claim',
        'amount_paid_on_contents_claim',
        'total_building_insurance_coverage',
        'total_contents_insurance_coverage',
    ], axis=1)
    .loc[lambda x: (
        (x['total_loss_ratio'] <= 1)
        & (x['loss_ratio_building'] <= 1)
        & (x['loss_ratio_content'] <= 1)

        & (x['latitude'] != 0.0)
        & (x['longitude'] != 0.0)
        & (x['elevation_difference'].between(-30.0, 30.0))
        & (x['rated_flood_zone'] != 'UNK')
        & (x['year_of_loss'] >= 1978)
        & (x['original_construction_date'] >= 1800)
        & (x['reported_zip_code'] > 100)
    )]
    .reset_index(drop=True)
)
data_cleaned.to_csv('FEMA_Data_Cleaned_Regression.csv', index=False)
