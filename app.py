from flask import Flask, request, jsonify
from classify import get_prediction

app = Flask(__name__)

@app.route("/predict-alphabet", methods = ["POST"])
def predict_data():
    img = request.files.get("alphabet")
    predict = get_prediction(img)
    return jsonify({
        "prediction": predict
    }, 200)