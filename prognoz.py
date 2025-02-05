import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_excel("./wb_orders.xlsx", engine="openpyxl")

cols_to_drop = ["Столбец 1", "Column 20", "warehouse_name", "oblast", "odid", 
                "category", "brand", "cancel_dt", "sticker", "order_type", 
                "created_at", "updated_at", "spp", "Столбец 19"]
df = df.drop(columns=cols_to_drop)

df = df[df['date'].apply(lambda x: pd.to_datetime(x, format='%d.%m.%Y', errors='coerce') is not pd.NaT)]
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df['month'] = df['date'].dt.month
df = df.sort_values('date')

df = df[df['is_cancel'] == False]

df['full_price'] = (df['total_price'] / (1 - df['discount_percent'].replace(100, 99.9)/100)).astype(int)

aggregated = df.groupby(['nm_id', 'month']).agg(
    total_revenue=('full_price', 'sum'),
    total_quantity=('full_price', 'count')
).reset_index()

all_nm_ids = aggregated['nm_id'].unique()
full_grid = pd.MultiIndex.from_product(
    [all_nm_ids, range(1, 13)],
    names=['nm_id', 'month']
).to_frame(index=False).reset_index(drop=True)

aggregated = full_grid.merge(
    aggregated,
    how='left',
    on=['nm_id', 'month']
).dropna()

aggregated = aggregated.sort_values(['nm_id', 'month'])

aggregated['month_sin'] = np.sin(2 * np.pi * aggregated['month']/12)
aggregated['month_cos'] = np.cos(2 * np.pi * aggregated['month']/12)

aggregated = aggregated.dropna()

features = ['nm_id', 'month_sin', 'month_cos']
X = aggregated[features]
y_revenue = aggregated['total_revenue'] 
y_quantity = aggregated['total_quantity']

model_revenue = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model_revenue.fit(X, y_revenue)

model_quantity = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model_quantity.fit(X, y_quantity)

january_data = []
for nm_id in all_nm_ids:
    product_data = aggregated[aggregated['nm_id'] == nm_id].sort_values('month')
    
    if len(product_data) == 0:
        continue
        
    january_row = {
        'nm_id': nm_id,
        'month': 1,
        'month_sin': np.sin(2 * np.pi * 1/12),
        'month_cos': np.cos(2 * np.pi * 1/12)
    }
    
    january_data.append(january_row)

january_df = pd.DataFrame(january_data)

predictions_revenue = model_revenue.predict(january_df[features])
predictions_quantity = model_quantity.predict(january_df[features])

result = pd.DataFrame({
    'nm_id': january_df['nm_id'],
    'forecast_revenue': predictions_revenue,
    'forecast_quantity': predictions_quantity
})

with pd.ExcelWriter("january_forecast_with_revenue_and_quantity.xlsx", engine="openpyxl") as writer:
   
    result.to_excel(writer, sheet_name='January Forecast', index=False)
    
    df.to_excel(writer, sheet_name='Original Data', index=False)
print("Прогноз успешно сохранен!")
print(f"Обработано товаров: {len(result)}")
print(f"Пример прогноза:\n{result.head()}")
