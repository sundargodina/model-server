from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("/Users/sundargodina/Downloads/catboost.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    print("Request received:", content)  # Debug print

    features = content["data"]["features"]
    if isinstance(features[0], float) or isinstance(features[0], int):
        features = [features]

    prediction = model.predict(features)
    print("Prediction result:", prediction)  # Debug print
    return jsonify({"predictions": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)