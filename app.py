import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))

# Load the scaler (assuming you saved it previously)
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(f"Received data: {data}")
    
    # Convert the input data to a numpy array and reshape it
    input_data = np.array(list(data.values())).reshape(1, -1)
    
    # Scale the input data
    new_data = scaler.transform(input_data)
    
    # Predict the result
    output = regmodel.predict(new_data)
    
    print(f"Prediction: {output[0]}")
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
