from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        final_features = scaler.transform([features])
        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = "⚠️ Heart Disease Detected"
            color = "red"
        else:
            result = "✅ No Heart Disease Detected"
            color = "green"

        return render_template("index.html", prediction_text=result, color=color)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: Invalid Input", color="red")

if __name__ == "__main__":
    app.run(debug=True)
