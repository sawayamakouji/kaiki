from flask import Flask, request, render_template, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

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
def predict_sales(model, start_date, end_date, df, feature_names):
    future_dates = pd.date_range(start=start_date, end=end_date)
    future_data = []
    conditions = []
    
    weathers = ['clear', 'rainy', 'snowy']
    promotions = ['none', 'discount', 'special']
    
    for date in future_dates:
        for weather in weathers:
            for promotion in promotions:
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
                conditions.append((date, weather, promotion))
    
    future_df = pd.DataFrame(future_data)
    future_df = future_df[feature_names]  # Ensure the columns are in the same order as in training
    predictions = model.predict(future_df)
    return predictions, conditions

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
    
    predictions, conditions = predict_sales(model, start_date, end_date, df, feature_names)
    
    results = []
    for (date, weather, promotion), prediction in zip(conditions, predictions):
        results.append({
            'date': str(date.date()),
            'weather': weather,
            'promotion': promotion,
            'predicted_sales': prediction
        })
    
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Pivot the DataFrame to have the desired format
    pivot_df = results_df.pivot_table(index='date', columns=['weather', 'promotion'], values='predicted_sales').reset_index()

    # Save the pivoted DataFrame to a CSV file
    csv_file = 'predictions.csv'
    pivot_df.to_csv(csv_file, index=False)

    return render_template('results.html', tables=[pivot_df.to_html(classes='data', index=False, header=False)], csv_file=csv_file)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)
 