from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load the model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r"@[A-Za-z0-9]+|#\S+|RT[\s]+|http\S+|[^a-zA-Z]", " ", text)  # Clean up text
    text = text.lower()
    
    # Tokenize, remove stopwords, and slang
    words = [word for word in text.split() if word not in stopwords.words("english") and word not in {"lol", "omg", "idk", "btw", "u", "r", "ur", "pls"}]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Prediction function
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess and predict
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]

    # Response based on prediction
    bullying_types = {
        "not_bullying": "This is not bullying.",
        "gender": "This is gender-based bullying.",
        "ethnicity": "This is ethnicity-based bullying.",
    }
    
    # Default fallback message if prediction type isn't recognized
    response_message = bullying_types.get(prediction, "This is bullying of message.")
    
    return jsonify({"prediction": response_message})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
