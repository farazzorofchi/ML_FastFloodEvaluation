import random

import pandas as pd
import numpy as np


data = pd.read_csv('openFEMA_claims20190831.csv')
data['amountpaidonincreasedcostofcomplianceclaim'] = (
    data['amountpaidonincreasedcostofcomplianceclaim'].fillna(0)
)
data['amountpaidonbuildingclaim'] = data['amountpaidonbuildingclaim'].fillna(0)
data['amountpaidoncontentsclaim'] = data['amountpaidoncontentsclaim'].fillna(0)
data['TIV'] = (
    data['totalbuildinginsurancecoverage'] +
    data['totalcontentsinsurancecoverage']
)
data['LOSS'] = (
    data['amountpaidonbuildingclaim'] +
    data['amountpaidoncontentsclaim']
)
data['loss_ratio_building'] = (
    data['amountpaidonbuildingclaim'] /
    data['totalbuildinginsurancecoverage']
)
data['loss_ratio_content'] = (
    data['amountpaidoncontentsclaim'] /
    data['totalcontentsinsurancecoverage']
)
data['loss_ratio_overall'] = data['LOSS'] / data['TIV']

# Keep only year for year built of the building
yearbuilt = []
for i in range(0, len(data)):
    try:
        yearbuilt.append(int(data['originalconstructiondate'][i][:4]))
    except:
        yearbuilt.append(random.randint(1970, 2019))
data['yearbuilt'] = yearbuilt

# Generalize Flood Zones:
FloodZone = []
for i in range(0, len(data)):
    try:
        if data['floodzone'][i][:2] == 'AO':
            FloodZone.append('AO')
        elif data['floodzone'][i][:2] == 'AH':
            FloodZone.append('AH')
        elif data['floodzone'][i][:2] == 'AR':
            FloodZone.append('AR')
        elif data['floodzone'][i][:2] == 'AE':
            FloodZone.append('AE')
        elif data['floodzone'][i][:2] == 'VE':
            FloodZone.append('VE')
        elif data['floodzone'][i][:3] == 'A99':
            FloodZone.append('A99')
        elif data['floodzone'][i][0] == 'A':
            if len(data['floodzone'][i][0]) > 2:
                FloodZone.append('AE')
            else:
                FloodZone.append('A')
        elif data['floodzone'][i][0] == 'V':
            if len(data['floodzone'][i][0]) > 2:
                FloodZone.append('VE')
            else:
                FloodZone.append('V')
        elif data['floodzone'][i][0] == 'X':
            FloodZone.append('X')
        elif data['floodzone'][i][0] == 'B':
            FloodZone.append('X')
        elif data['floodzone'][i][0] == 'C':
            FloodZone.append('X')
        else:
            FloodZone.append('UNK')
    except:
        FloodZone.append('UNK')
data['floodzone'] = FloodZone

zipcode = []
for i in range(0, len(data)):
    try:
        if int(data['reportedzipcode'][i]) > 100:
            zipcode.append(int(data['reportedzipcode'][i]))
        else:
            zipcode.append(-999999)
    except:
        zipcode.append(-999999)
data['ZipCode'] = zipcode

for i in range(0, len(data)):
    if (
        (not np.isnan(data['lowestfloorelevation'][i])) &
        (not np.isnan(data['basefloodelevation'][i]))
    ):
        data['elevationdifference'][i] = (
            data['lowestfloorelevation'][i] -
            data['basefloodelevation'][i]
        )

data = data[data['elevationdifference'] != 999]
data.reset_index(drop=True, inplace=True)
data.drop('asofdate', axis=1, inplace=True)
data.drop('countycode', axis=1, inplace=True)
data.drop('censustract', axis=1, inplace=True)
data.drop('reportedcity', axis=1, inplace=True)
data.drop('dateofloss', axis=1, inplace=True)
data.drop('elevationcertificateindicator', axis=1, inplace=True)
data.drop('lowestadjacentgrade', axis=1, inplace=True)
data.drop('lowestfloorelevation', axis=1, inplace=True)
data.drop('basefloodelevation', axis=1, inplace=True)
data.drop('originalconstructiondate', axis=1, inplace=True)
data.drop('originalnbdate', axis=1, inplace=True)
data.drop('amountpaidonincreasedcostofcomplianceclaim', axis=1, inplace=True)
data.drop('ratemethod', axis=1, inplace=True)
data.drop('smallbusinessindicatorbuilding', axis=1, inplace=True)
data.drop('state', axis=1, inplace=True)
data.drop('reportedzipcode', axis=1, inplace=True)
data.drop('primaryresidence', axis=1, inplace=True)
print('len(data) before cleaning =', len(data))

delete_row = data[np.isnan(data['latitude'])].index
data = data.drop(delete_row)
delete_row = data[np.isnan(data['longitude'])].index
data = data.drop(delete_row)
delete_row = data[np.isnan(data['numberoffloorsintheinsuredbuilding'])].index
data = data.drop(delete_row)

# Keeping only up to 30 feet difference
data = data[
    (data['elevationdifference'] <= 30) &
    (data['elevationdifference'] >= -30)
]
data = data[data['yearbuilt'] >= 1800]
data = data[data['nonprofitindicator'] != '0']
data = data[data['ZipCode'] != -999999]
data = data[data['TIV'] != 0]
data = data[data['obstructiontype'] != '*']
data['obstructiontype'] = pd.to_numeric(data['obstructiontype'])
data['obstructiontype'] = data['obstructiontype'].astype(str)
data['obstructiontype'].fillna('UNK', inplace=True)
data['obstructiontype'][data['obstructiontype'] == 'nan'] = 'UNK'
data['agriculturestructureindicator'].fillna('N', inplace=True)
data['basementenclosurecrawlspacetype'].fillna(0, inplace=True)
data['condominiumindicator'].fillna('N', inplace=True)
data['policycount'].fillna(1, inplace=True)
data['crsdiscount'].fillna(0, inplace=True)
data['elevatedbuildingindicator'].fillna('UNK', inplace=True)
data['floodzone'].fillna('UNK', inplace=True)
data['houseworship'].fillna('N', inplace=True)
data['locationofcontents'].fillna('UNK', inplace=True)
data['nonprofitindicator'].fillna('N', inplace=True)
data['occupancytype'].fillna(1.0, inplace=True)
data['postfirmconstructionindicator'].fillna('UNK', inplace=True)
data = data[data['floodzone'] != 'UNK']
data.reset_index(drop=True, inplace=True)
print('len(data) after cleaning =', len(data))

data.to_csv('FEMA_Data_Cleaned_Regression.csv', index=False)
