# Import necessary libraries
import pandas as pd

# Load the dataset
df = pd.read_pickle("merged_training.pkl")

# Preview the first 50 rows of the dataset
print(df[:50])

# Plot the distribution of emotions in the dataset
df['emotions'].value_counts().plot(kind='bar', title='Emotion Distribution')

"""
Emotion mapping:
anger: 0
fear: 1
joy: 2
love: 3
sadness: 4
surprise: 5
"""

# Check for duplicate and missing data
print("Number of duplicate rows:", df.duplicated().sum())
print("Number of missing values in each column:", df.isnull().sum())

# Import NLTK library for text processing
import nltk
from nltk.corpus import stopwords

# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define a text cleaning function
def clean_text(text):
    """
    Clean the input text by:
    1. Converting to lowercase.
    2. Removing non-alphanumeric characters.
    3. Removing stopwords.
    """
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply the cleaning function to the 'text' column
df['clean_text'] = df['text'].apply(clean_text)

# Encode the emotion labels into numeric format
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['emotion_label'] = le.fit_transform(df['emotions'])  # Convert emotion labels to numeric

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['emotion_label'], test_size=0.2, random_state=42
)

# Transform the text data into TF-IDF feature vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit and transform training data
X_test_tfidf = vectorizer.transform(X_test)  # Transform testing data

# Train a logistic regression model for emotion detection
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)  # Fit the model to the training data

# Make predictions on the test data
predictions = model.predict(X_test_tfidf)

# Evaluate the model's performance using a classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions, target_names=le.classes_))

# Save the trained model and vectorizer for later use
import joblib

joblib.dump(model, 'emotion_model.pkl')  # Save the trained model
joblib.dump(vectorizer, 'emotion_vectorizer.pkl')  # Save the vectorizer

