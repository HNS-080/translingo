from flask import Flask, request, jsonify
import requests
import pandas as pd
import spacy
from dotenv import load_dotenv
from flask_cors import CORS
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load English NLP model
os.system("python -m spacy download en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

csv_path = os.path.join(os.path.dirname(__file__), "ComputerScience_Jargon_Dataset.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

# Load dataset and convert it into a dictionary (Term → Definition)
df = pd.read_csv(csv_path)

print("CSV Loaded Successfully! First few rows:")
print(df.head())  # Debugging output

definitions = dict(zip(df['TERMS'], df['Definition']))

# Hugging Face API details for text simplification
API_URL1 = "https://api-inference.huggingface.co/models/JexCaber/TransLingo"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}

@app.route("/simplify-text", methods=['POST'])
def simplify_text():
    try:
        data = request.json
        text = data.get('text')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        generation_params = {"inputs": text, "parameters": {"max_length": 150}}
        response = requests.post(API_URL1, headers=HEADERS, json=generation_params)
        
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route("/term-detection", methods=['POST'])
def term_detection():
    try:
        data = request.json
        text = data.get('text')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Process text with spaCy
        doc = nlp(text)
        extracted_terms = {}

        for token in doc:
            term = token.text
            if term in definitions and term not in extracted_terms:  # Check if it's a known term
                extracted_terms[term] = definitions[term]

        return jsonify({"extracted_terms": extracted_terms})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Welcome to the TransLingo API!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
