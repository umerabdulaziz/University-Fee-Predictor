
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from train_model import preprocessor
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
model = tf.keras.models.load_model("model.keras")
preprocessor = joblib.load("preprocessor.pkl")
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "predictions.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    country = db.Column(db.String(100))
    degree = db.Column(db.String(10))
    field = db.Column(db.String(50))
    gpa = db.Column(db.Float)
    uni_type = db.Column(db.String(20))
    predicted_fee = db.Column(db.Float)
with app.app_context():
    db.create_all()
@app.route('/api/predict', methods=["POST"])

def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    X = preprocessor.transform(input_df)

    prediction = model.predict(X)[0][0]
    prediction = round(prediction, 2)
    return jsonify({
        "predicted_fee": prediction
    })

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/predict', methods=["POST"])
def predict_form():
    degree = request.form['degree']
    field = request.form['field']
    country = request.form['country']
    gpa = float(request.form['gpa'])
    uni_type = request.form['uni_type']

    data = {
        "Country": country,
        "Degree": degree,
        "University_Type": uni_type,
        "Field": field,
        "GPA": gpa
    }
    input_df = pd.DataFrame([data])
    X = preprocessor.transform(input_df)
    prediction = model.predict(X)[0][0]
    prediction = round(prediction, 2)
    new_prediction = Prediction(
        degree=degree,
        field=field,
        country=country,
        gpa=gpa,
        uni_type=uni_type,
        predicted_fee=prediction
    )
    db.session.add(new_prediction)
    db.session.commit()

    return f"<h2>Predicted University Fee: ${prediction}</h2>"
if __name__ == '__main__':
    app.run(debug=True)