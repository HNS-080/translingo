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

definitions = dict(zip(df['TERMS'], df['Definition']))

# Hugging Face API details for text simplification
API_URL1 = "https://api-inference.huggingface.co/models/pszemraj/long-t5-tglobal-base-sci-simplification"
API_URL2 = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}
HEADERS2 = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN2')}"}


def nested_sliding_window_split(text, min_size=3, max_size=12, step_size=5):
    """Splits text into overlapping word chunks of varying sizes."""
    words = text.split()
    chunks = []
    
    for size in range(min_size, max_size + 1, step_size):
        for i in range(0, len(words) - size + 1, step_size):
            chunk = " ".join(words[i:i+size])
            chunks.append(chunk)
    
    return chunks


def clean_term_output(output_text):
    """Cleans extracted term and removes redundant patterns."""
    output_text = output_text.replace("Term: Term:", "Term:").replace("Term:", "", 1).strip()
    return output_text


def extract_terms_from_paragraph(paragraph):
    """Uses Hugging Face API to detect terms in text."""
    chunks = nested_sliding_window_split(paragraph)
    extracted_terms = {}

    for chunk in chunks:
        generation_params = {
            "inputs": chunk,
            "parameters": {"max_length": 512}
        }

        try:
            response = requests.post(API_URL2, headers=HEADERS2, json=generation_params)
            response.raise_for_status()
            output_text = response.json()[0]['generated_text']

            term, definition = output_text.split("| Definition: ", 1) if "| Definition: " in output_text else (output_text, "")
            term = clean_term_output(term).lower()

            if term and term not in extracted_terms:
                extracted_terms[term] = definition.strip()

        except requests.exceptions.RequestException as e:
            return {"error": "Hugging Face API request failed", "details": str(e)}

    return extracted_terms


@app.route("/simplify-text", methods=['POST'])
def simplify_text():
    try:
        data = request.json
        text = data.get('text')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Define generation parameters
        generation_params = {
            "inputs": text,
            "parameters": {
                "max_length": 150
            }
        }

        # Send request to Hugging Face API
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

        # Process text with spaCy first
        doc = nlp(text)
        extracted_terms = {}

        for token in doc:
            term = token.text
            if term in definitions and term not in extracted_terms:
                extracted_terms[term] = definitions[term]

        # If no terms detected, fallback to Hugging Face API extraction
        if not extracted_terms:
            extracted_terms = extract_terms_from_paragraph(text)

        return jsonify({"extracted_terms": extracted_terms})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Welcome to the TransLingo API!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
