from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import folium
from folium.plugins import HeatMap

app = Flask(__name__)

# Load the datasets
data = pd.read_csv('./Crop_recommendation.csv')
soil_data = pd.read_csv('./soil.csv')

# Ensure column names are stripped of any leading/trailing spaces
soil_data.columns = soil_data.columns.str.strip()

# Check if necessary columns are present
if 'latitude' not in soil_data.columns or 'longitude' not in soil_data.columns:
    raise ValueError("Required columns 'latitude' and 'longitude' are missing from soil_data.")

# Model and Scaler setup
X = data.iloc[:, :-1]  # Input features (N, P, K, temperature, humidity, pH, rainfall)
y = data.iloc[:, -1]   # Labels (crop names)

# Standardize the features
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract input data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Prepare the feature list
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scale the input features
        scaled_features = sc.transform(single_pred)

        # Predict the crop
        prediction = model.predict(scaled_features)

        
        result = "{} is the best crop to be cultivated right there".format(prediction[0])

        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

@app.route("/map", methods=['POST'])
def map_view():
    try:
        lat = float(request.form['latitude'])
        lon = float(request.form['longitude'])

        # Create a Folium map centered around the provided latitude and longitude
        m = folium.Map(location=[lat, lon], zoom_start=10)

        # Add a heat map layer
        if 'latitude' in soil_data.columns and 'longitude' in soil_data.columns:
            heat_data = [[row['latitude'], row['longitude']] for index, row in soil_data.iterrows()]
            HeatMap(heat_data).add_to(m)
        else:
            return render_template('index.html', result="Latitude and Longitude columns are missing in soil data.")

        # Save the map to an HTML string
        map_html = m._repr_html_()

        return render_template('index.html', map_html=map_html)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

@app.route("/state_analysis", methods=['POST'])
def state_analysis():
    try:
        state_name = request.form['state_name'].strip().title()

        # Reverse geocode to match state name with latitude and longitude from the CSV
        state_data = []
        for index, row in soil_data.iterrows():
            response = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={row['latitude']}&lon={row['longitude']}&zoom=5")
            data = response.json()
            if 'address' in data and 'state' in data['address']:
                if data['address']['state'].lower() == state_name.lower():
                    state_data.append(row)

        if not state_data:
            result = f"No data available for {state_name}. Please check the state name."
            return render_template('index.html', state_result=result)

        # Convert to DataFrame for easier processing
        state_df = pd.DataFrame(state_data)

        # Return the data found
        result = f"Data found for {state_name}: {state_df.to_html()}"

        return render_template('index.html', state_result=result)
    except Exception as e:
        return render_template('index.html', state_result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
