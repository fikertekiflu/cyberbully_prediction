from flask import Flask, request, jsonify
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

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
    text = re.sub(r"@[A-Za-z0-9]+", "", text)  # Remove @mentions
    text = re.sub(r"#\S+", "", text)  # Remove hashtags
    text = re.sub(r"RT[\s]+", "", text)  # Remove RT
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphanumeric characters
    text = text.lower()

    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]

    # Additional stopwords/slang removal
    slang_terms = ["lol", "omg", "idk", "btw", "u", "r", "ur", "pls"]
    words = [word for word in words if word not in slang_terms]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Prediction function
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]

    # Generate warning message based on prediction
    if prediction == "not_bullying":
        return jsonify({"prediction": "This is not bullying."})
    else:
        if prediction == "gender":
            return jsonify({"prediction": "This is gender-based bullying."})
        elif prediction == "ethnicity":
            return jsonify({"prediction": "This is ethnicity-based bullying."})
        else:
            return jsonify({"prediction": "This is bullying of message."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
