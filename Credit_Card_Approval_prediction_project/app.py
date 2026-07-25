from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("credit_card_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]

    if prediction == 1:
        result = "Credit Card Approved"
    else:
        result = "Credit Card Rejected"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)