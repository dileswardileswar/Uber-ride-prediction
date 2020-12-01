import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open(r'F:\8.End_to_End_Project\1.Uber Ride weekly Analisys\MY_Own_Code\taxi.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(value) for value in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = int(prediction[0])
    return render_template('index.html', prediction_text="Number of Weekely Rides should be {}".format(output))





if __name__ == '__main__':
    app.run()