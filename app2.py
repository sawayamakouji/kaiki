import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ダミーデータの生成
def generate_dummy_data():
    # 日付の範囲を設定
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start_date, end_date)

    # ダミーデータのリストを生成
    data = []
    for date in date_range:
        store_name = 'Store A'
        product_name = 'Product 1'
        sales_date = date.strftime('%Y-%m-%d')
        sales_quantity = np.random.randint(50, 200)
        sales_amount = sales_quantity * np.random.randint(100, 500)
        weather = np.random.choice(['clear', 'rainy', 'snowy'])
        promotion = np.random.choice(['none', 'discount', 'special'])
        
        data.append([store_name, product_name, sales_date, sales_quantity, sales_amount, weather, promotion])

    # DataFrameに変換
    df = pd.DataFrame(data, columns=['store_name', 'product_name', 'sales_date', 'sales_quantity', 'sales_amount', 'weather', 'promotion'])
    
    return df

# ダミーデータの生成
dummy_data = generate_dummy_data()

# CSVファイルとして保存
file_path = 'dummy_sales_data.csv'
dummy_data.to_csv(file_path, index=False)

print(f"CSVファイルが生成されました: {file_path}")
