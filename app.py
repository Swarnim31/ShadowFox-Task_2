import os
import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- This function trains the model if it doesn't exist ---
# It's the best way to ensure your app and model are perfectly in sync.
def train_model_if_needed(model_path='car_price_model.pkl', data_path='car.csv'):
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Training a new one from '{data_path}'...")
        # Load data and calculate car age for training
        car_data = pd.read_csv(data_path)
        current_year = datetime.now().year
        car_data['Car_Age'] = current_year - car_data['Year']
        
        # Select and encode features exactly as the model will need them
        final_df = car_data[['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Selling_Price']]
        final_df = pd.get_dummies(final_df, drop_first=True)
        
        # Define features (X) and target (y)
        X = final_df.drop('Selling_Price', axis=1)
        y = final_df['Selling_Price']
        
        # Train the model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # IMPORTANT: Save both the model and the column order it was trained on
        with open(model_path, 'wb') as f:
            pickle.dump((rf_model, X.columns), f)
        print("Model trained and saved.")

# --- Flask App ---
app = Flask(__name__)

# Train the model if needed when the app starts
train_model_if_needed()

# Load the trained model and its column structure
try:
    model, model_columns = pickle.load(open('car_price_model.pkl', 'rb'))
except FileNotFoundError:
    model, model_columns = None, None # Handle case where training fails

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    error_message = None

    if request.method == 'POST':
        # --- DEBUGGING LINE ---
        # This will print the exact data received from the form to your terminal.
        print(f"Form data received: {request.form}")

        if not model:
            error_message = "Model is not loaded. Please check the server."
            return render_template('index.html', error=error_message)

        try:
            # --- Get Data from Form ---
            # CORRECTED: Keys are now TitleCase to match the HTML form's 'name' attributes.
            present_price = float(request.form['Present_Price'])
            kms_driven = int(request.form['Kms_Driven'])
            owner = int(request.form['Owner'])
            car_age = int(request.form['Car_Age'])
            fuel_type = request.form['Fuel_Type']
            seller_type = request.form['Seller_Type']
            transmission = request.form['Transmission']

            # --- Prepare Features for Prediction ---
            input_data = pd.DataFrame(
                [[present_price, kms_driven, owner, car_age,
                  1 if fuel_type == 'Diesel' else 0,
                  1 if fuel_type == 'Petrol' else 0,
                  1 if seller_type == 'Individual' else 0,
                  1 if transmission == 'Manual' else 0]],
                columns=['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age', 
                         'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 
                         'Seller_Type_Individual', 'Transmission_Manual']
            )
            
            # Ensure the order of columns matches the model's training data
            input_data = input_data.reindex(columns=model_columns, fill_value=0)

            # --- Make Prediction ---
            prediction = model.predict(input_data)
            predicted_price = round(prediction[0], 2)

        except Exception as e:
            # This block catches the error and prevents the app from crashing.
            print(f"Error during prediction: {e}")
            error_message = "Error in input. Please check the values and try again."
    
    return render_template('index.html', predicted_price=predicted_price, error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
