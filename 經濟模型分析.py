# 經濟政策影響分析模型

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# 讀取三份資料
tariff_df = pd.read_csv("API_TM.TAX.MRCH.SM.AR.ZS_DS2_en_csv.csv", skiprows=4)
import_value_df = pd.read_csv("API_TM.VAL.MRCH.CD.WT_DS2_en_csv.csv", skiprows=4)
import_gdp_df = pd.read_csv("API_NE.IMP.GNFS.ZS_DS2_en_csv.csv", skiprows=4)

# 篩選出美國資料
tariff_us = tariff_df[tariff_df['Country Name'] == 'United States']
import_value_us = import_value_df[import_value_df['Country Name'] == 'United States']
import_gdp_us = import_gdp_df[import_gdp_df['Country Name'] == 'United States']

# 取共同年份（1989–2022）
years = [str(y) for y in range(1989, 2023)]
valid_years = list(set(years) & set(tariff_us.columns) & set(import_value_us.columns) & set(import_gdp_us.columns))
valid_years = sorted(valid_years)

# 整合為單一 DataFrame
full_data = pd.DataFrame({
    'Year': valid_years,
    'Tariff Rate (%)': tariff_us[valid_years].iloc[0].values.astype(float),
    'Import Value (US$)': import_value_us[valid_years].iloc[0].values.astype(float),
    'Imports % of GDP': import_gdp_us[valid_years].iloc[0].values.astype(float)
})
full_data.dropna(inplace=True)


# 線性回歸模型訓練與視覺化
X = full_data[['Tariff Rate (%)']].values
y_value = full_data['Import Value (US$)'].values
y_gdp = full_data['Imports % of GDP'].values

model_value = LinearRegression().fit(X, y_value)
model_gdp = LinearRegression().fit(X, y_gdp)

X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_pred_value = model_value.predict(X_range)
y_pred_gdp = model_gdp.predict(X_range)

plt.figure(figsize=(10, 6))
plt.scatter(full_data['Tariff Rate (%)'], y_value, label='Import Value (US$)', color='royalblue', alpha=0.7)
plt.plot(X_range, y_pred_value, color='blue', linestyle='--', label='Regression Line (Value)')

plt.scatter(full_data['Tariff Rate (%)'], y_gdp * 1e12, label='Imports % of GDP (scaled)', color='darkgreen', alpha=0.7)
plt.plot(X_range, y_pred_gdp * 1e12, color='green', linestyle='--', label='Regression Line (GDP %)')

plt.title("Tariff Rate vs Import Metrics (US, 1989–2022)")
plt.xlabel("Tariff Rate (%)")
plt.ylabel("Value / Scaled GDP Share")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 趨勢視覺化含政策年份
years = full_data['Year'].astype(int)
tariff = full_data['Tariff Rate (%)']
import_value = full_data['Import Value (US$)'] / 1e12
import_gdp = full_data['Imports % of GDP']

plt.figure(figsize=(12, 6))
plt.plot(years, tariff, label="Tariff Rate (%)", color="orange", linewidth=2)
plt.plot(years, import_value, label="Import Value (Trillion US$)", color="royalblue", linewidth=2)
plt.plot(years, import_gdp, label="Imports % of GDP", color="green", linewidth=2)

for y in [1994, 2001, 2008, 2018, 2020]:
    plt.axvline(x=y, color='gray', linestyle='--', alpha=0.5)
    plt.text(y + 0.1, plt.ylim()[1] * 0.95, str(y), rotation=90, verticalalignment='top', fontsize=9)

plt.title("US Trade Policy & Market Trend (1989–2022)")
plt.xlabel("Year")
plt.ylabel("Metric")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
