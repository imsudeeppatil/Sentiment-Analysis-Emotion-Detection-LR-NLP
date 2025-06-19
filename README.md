# Sentiment Analysis and Emotion Detection from Text

This mini project uses **Machine Learning** and **Natural Language Processing (NLP)** to analyze text and determine both sentiment (positive, negative, or neutral) and emotion (joy, anger, sadness, etc.). The application is built with **Python**, **Flask**, and a set of ML models using **Logistic Regression**.

## 📌 Project Objectives

- Classify textual data into sentiment categories: Positive, Negative, Neutral.
- Detect and classify specific emotions like Joy, Anger, Sadness, etc.
- Handle real-world text challenges like sarcasm, mixed emotions, and noisy input.
- Build a scalable and user-friendly web interface.
- Enable speech-to-text for accessibility.

## 🛠️ Tools & Technologies

- **Languages & Frameworks**: Python, Flask, HTML/CSS/JS
- **Libraries**: scikit-learn, NLTK, pandas, matplotlib, joblib
- **Development Tools**: Jupyter Notebook, VSCode, Anaconda, GitHub
- **Deployment**: Flask Web Server

## 📂 Project Structure

project/
├── templates/
│ └── index.html # Web interface
├── model.pkl # Sentiment analysis model
├── emotion_model.pkl # Emotion detection model
├── vectorizer.pkl # Sentiment TF-IDF vectorizer
├── emotion_vectorizer.pkl # Emotion TF-IDF vectorizer
├── app.py # Flask backend
├── Sentiment_Analysis.py # Script for training sentiment model
├── Emotion_Detection.py # Script for training emotion model
└── merged_training.pkl # Dataset for emotion detection


## 🧠 ML Models Used

### Sentiment Analysis:
- **Algorithm**: Logistic Regression
- **Input**: Text (processed with TF-IDF)
- **Output**: Sentiment Score + Category

### Emotion Detection:
- **Algorithm**: Logistic Regression (Multiclass)
- **Emotions**: Joy, Anger, Sadness, Fear, Love, Surprise

## 🖼️ Features

- TF-IDF vectorization for feature extraction
- Speech-to-Text input support
- Flask-based web interface with theme toggle
- Real-time bar graph showing emotion distribution
- Score interpretation for sentiments
- Saved models for quick deployment using `joblib`

## 🚀 How to Run

1. **Install dependencies**:
   pip install -r requirements.txt
**2.** **Train the models**:
   python Sentiment_Analysis.py
   python Emotion_Detection.py
**3.** **Run Flask Application**:
   python app.py
**4.** Visit: http://localhost:5000 in your browser.

## 📈 Results
Sentiment Accuracy: ~88%
Emotion Accuracy (Weighted F1-Score): ~90%
Best performance for Sadness and Joy, with areas to improve for Love and Surprise.

## 🧩 Future Scope
- Integrate deep learning models (BERT, LSTM).
- Expand to multilingual support.
- Improve performance on underrepresented emotions.
- Enable advanced visualization using Plotly or D3.js.

## 👨‍💻 Contributors
- Sneha R
- Sudeep Patil 
- Thushar DM
- Vinayak Rajput
  
Guided by: Prof. Sunanda H G, Assistant Professor, Dept. of CSE, BIT
