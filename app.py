# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("advertising_lr_model.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            tv = float(request.form.get("tv", 0))
            radio = float(request.form.get("radio", 0))
            newspaper = float(request.form.get("newspaper", 0))
            X = np.array([[tv, radio, newspaper]])
            pred = model.predict(X)[0]
            prediction = round(float(pred), 3)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
