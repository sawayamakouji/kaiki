from flask import Flask, request, render_template, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
import io

app = Flask(__name__)

# Load the data and preprocess it
def preprocess_data(file):
    df = pd.read_csv(file)
    df['sales_date'] = pd.to_datetime(df['sales_date'])
    df['day_of_week'] = df['sales_date'].dt.dayofweek
    df['month'] = df['sales_date'].dt.month
    df['day'] = df['sales_date'].dt.day

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['weather', 'promotion'])

    return df

# Train the regression model
def train_model(df):
    X = df.drop(columns=['sales_quantity', 'sales_amount', 'sales_date', 'store_name', 'product_name'])
    y = df['sales_quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X.columns  # return the model and the feature names

# Predict future sales
def predict_sales(model, start_date, end_date, df, feature_names, store_name, product_name, weather, promotion):
    future_dates = pd.date_range(start=start_date, end=end_date)
    future_data = []
    for date in future_dates:
        day_of_week = date.weekday()
        month = date.month
        day = date.day

        data = {'day_of_week': day_of_week, 'month': month, 'day': day,
                'weather_clear': 1 if weather == 'clear' else 0,
                'weather_rainy': 1 if weather == 'rainy' else 0,
                'weather_snowy': 1 if weather == 'snowy' else 0,
                'promotion_none': 1 if promotion == 'none' else 0,
                'promotion_discount': 1 if promotion == 'discount' else 0,
                'promotion_special': 1 if promotion == 'special' else 0}

        future_data.append(data)
    
    future_df = pd.DataFrame(future_data)
    future_df = future_df[feature_names]  # Ensure the columns are in the same order as in training
    predictions = model.predict(future_df)
    return predictions, future_dates

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    df = preprocess_data(file)
    model, feature_names = train_model(df)

    start_date = request.form['start_date']
    end_date = request.form['end_date']
    store_name = request.form['store_name']
    product_name = request.form['product_name']
    weather = request.form['weather']
    promotion = request.form['promotion']
    
    predictions, future_dates = predict_sales(model, start_date, end_date, df, feature_names, store_name, product_name, weather, promotion)
    
    results = {str(date.date()): prediction for date, prediction in zip(future_dates, predictions)}
    
    return render_template('results.html', results=results, start_date=start_date, end_date=end_date, store_name=store_name, product_name=product_name, weather=weather, promotion=promotion)

@app.route('/download_csv')
def download_csv():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    store_name = request.args.get('store_name')
    product_name = request.args.get('product_name')
    weather = request.args.get('weather')
    promotion = request.args.get('promotion')

    # 再度データを読み込み、予測を行う
    df = preprocess_data(request.files['file'])
    model, feature_names = train_model(df)
    predictions, future_dates = predict_sales(model, start_date, end_date, df, feature_names, store_name, product_name, weather, promotion)
    
    results = pd.DataFrame({
        "Date": [date.date() for date in future_dates],
        "Prediction": predictions
    })

    csv_data = results.to_csv(index=False)
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype="text/csv",
        as_attachment=True,
        attachment_filename="predictions.csv"
    )

if __name__ == '__main__':
    app.run(debug=True)
