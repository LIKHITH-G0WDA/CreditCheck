from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load('credit_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        if file and file.filename.endswith('.csv'):
            # Read the CSV file
            try:
                df = pd.read_csv(file)
            except Exception as e:
                return render_template('index.html', error=f"Error reading the file: {e}")
            
            # Prepare the data for prediction
            feature_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                               'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
                               'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

            # Check for capitalized 'ID' column
            id_column = 'ID' if 'ID' in df.columns else 'id'
            if id_column not in df.columns:
                return render_template('index.html', error="CSV file must contain an 'id' or 'ID' column.")
            
            # Check for required features
            if set(feature_columns).issubset(df.columns):
                ids = df[id_column]
                X = df[feature_columns]
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                df['Prediction'] = ['High Risk' if pred == 1 else 'Low Risk' for pred in predictions]

                # Select only 'id' and 'Prediction' columns for display
                results_df = df[[id_column, 'Prediction']]

                # Convert DataFrame to HTML table
                results_table = results_df.to_html(classes='data', index=False)
                return render_template('index.html', tables=[results_table], titles=results_df.columns.values)
            else:
                return render_template('index.html', error="CSV file does not contain all required features.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run()#host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
