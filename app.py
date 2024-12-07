from flask import Flask, request, jsonify, render_template
import joblib
import pickle
from collections import Counter
import string
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the trained sentiment analysis model and vectorizer
classifier = joblib.load('model.pkl')
tfidf_vectorizer = joblib.load('vectorizer.pkl')

# Load the emotion detection model
with open("emotion_model.pkl", "rb") as emotion_file:
    emotion_model = pickle.load(emotion_file)

print("Emotion model loaded with keys:", list(emotion_model.keys())[:10])  # Debug print

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.json.get("text", "")
    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    print("Raw input text:", input_text)  # Debug print

    # Preprocess text
    words = input_text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    print("Processed words:", words)  # Debug print

    # Lemmatize words to match base form
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    print("Lemmatized words:", lemmatized_words)  # Debug print

    # Match words with emotions
    detected_emotions = [emotion_model.get(word, None) for word in lemmatized_words]
    detected_emotions = [emotion for emotion in detected_emotions if emotion is not None]
    print("Detected emotions:", detected_emotions)  # Debug print

    # Count emotions
    emotion_counter = Counter(detected_emotions)
    print("Emotion counter:", emotion_counter)  # Debug print

    # Sentiment analysis
    input_features = tfidf_vectorizer.transform([input_text])
    y_proba = classifier.predict_proba(input_features)
    positive_score = y_proba[0][1]
    sentiment = "Neutral"
    if positive_score < 0.4:
        sentiment = "Negative"
    elif positive_score > 0.6:
        sentiment = "Positive"

    # Debug: print out what we are sending to the frontend
    print("Returning response:", {
        "text": input_text,
        "sentiment": {"score": positive_score, "sentiment": sentiment},
        "emotions": dict(emotion_counter)
    })

    return jsonify({
        "text": input_text,
        "sentiment": {"score": positive_score, "sentiment": sentiment},
        "emotions": dict(emotion_counter)
    })

if __name__ == "__main__":
    app.run(debug=True)
