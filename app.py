import numpy as np

from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
std=StandardScaler()

app = Flask(__name__)
model = pickle.load(open('rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print(request.form)
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output=round(prediction[0], 2)
    return render_template('index.html',
                           prediction_text='Your Shoe Price would be : {:.2f} '.format(output))

if __name__ == "__main__":
    app.run(debug=True)