import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(r"openFEMA_claims20190831.csv")
temp = df.groupby(['yearofloss'])[['amountpaidonbuildingclaim']].count()
temp.reset_index(inplace=True)
temp2 = temp[temp['amountpaidonbuildingclaim'].notna()]
x = temp2['yearofloss']
y = temp2['amountpaidonbuildingclaim']

plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.xlabel('year', fontsize=18)
plt.ylabel('count of paid building claims', fontsize=16)
plt.plot(x, p(x), "r--")
plt.xticks(np.arange(min(x), max(x)+1, 2.0))
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()

df['LossRatio'] = (
    (df['amountpaidonbuildingclaim'] + df['amountpaidoncontentsclaim']) /
    (df['totalbuildinginsurancecoverage'] + df['totalcontentsinsurancecoverage'])
)
temp = df.groupby(['yearofloss'])[['LossRatio']].mean()
temp.reset_index(inplace=True)
temp2 = temp[temp['LossRatio'] <= 1]
x = temp2['yearofloss']
y = temp2['LossRatio']

plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.xlabel('year', fontsize=18)
plt.ylabel('Average Loss Ratio', fontsize=16)
plt.plot(x, p(x), "r--")
plt.xticks(np.arange(min(x), max(x)+1, 2.0))
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
