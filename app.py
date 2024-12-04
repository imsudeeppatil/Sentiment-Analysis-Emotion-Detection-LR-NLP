from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer from your existing script
# Assuming you have saved them as 'model.pkl' and 'vectorizer.pkl'
classifier = joblib.load('model.pkl')
tfidf_vectorizer = joblib.load('vectorizer.pkl')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.json.get("text", "")
    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    # Transform the input text and predict probabilities
    input_features = tfidf_vectorizer.transform([input_text])
    y_proba = classifier.predict_proba(input_features)
    positive_score = y_proba[0][1]

    # Interpret the score into sentiment
    if positive_score < 0.4:
        sentiment = "Negative"
    elif 0.4 <= positive_score <= 0.6:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    return jsonify({"text": input_text, "score": positive_score, "sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
