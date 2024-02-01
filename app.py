from flask import Flask, request, jsonify
import pickle
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True

from flask_cors import CORS

CORS(app)

# Load the model
with open('crime_records.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    print(model)
# model = pickle.load(open('crime_records.pkl', 'rb'))
model_columns = joblib.load('crime_records.pkl')


@app.route('/')
def default():
    return '<h1> API server is Working </h1>'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # category = ['AREA', 'Month', 'Day', 'Hour']
    # query = pd.get_dummies(data, columns=category, dummy_na=True)
    input_data = [data.get('AREA'), data.get('Month'), data.get('Day'), data.get('Hour')]
    # print(query)
    input_array = np.array(input_data).reshape(1, -1)
    result = model.predict(input_array)
    print(result)
   
    return (jsonify({'prediction' : result.tolist()}))
    # return jsonify(result)

    # return str(loan_predict)
    # return '<h1> Predicting... </h1>'


if __name__ == '__main__':
    app.run(port=5000, debug=True)