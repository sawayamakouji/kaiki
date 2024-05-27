from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime

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
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sales Prediction</title>
    </head>
    <body>
        <h1>Upload Sales Data and Predict Future Sales</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label for="file">Upload CSV:</label>
            <input type="file" id="file" name="file" accept=".csv" required><br><br>

            <label for="start_date">Start Date (YYYY-MM-DD):</label>
            <input type="date" id="start_date" name="start_date" required><br><br>

            <label for="end_date">End Date (YYYY-MM-DD):</label>
            <input type="date" id="end_date" name="end_date" required><br><br>

            <h2>Enter New Data:</h2>

            <label for="store_name">Store Name:</label>
            <input type="text" id="store_name" name="store_name" required><br><br>

            <label for="product_name">Product Name:</label>
            <input type="text" id="product_name" name="product_name" required><br><br>

            <label for="weather">Weather:</label>
            <select id="weather" name="weather" required>
                <option value="clear">Clear</option>
                <option value="rainy">Rainy</option>
                <option value="snowy">Snowy</option>
            </select><br><br>

            <label for="promotion">Promotion:</label>
            <select id="promotion" name="promotion" required>
                <option value="none">None</option>
                <option value="discount">Discount</option>
                <option value="special">Special</option>
            </select><br><br>

            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    """
    return render_template_string(html_content)

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
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
