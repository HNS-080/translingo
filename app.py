from flask import Flask, request, jsonify  # Import Flask, request, and jsonify
import requests
import os
print("HUGGINGFACE_TOKEN:", os.getenv('HUGGINGFACE_TOKEN'))
print("HUGGINGFACE_TOKEN2:", os.getenv('HUGGINGFACE_TOKEN2'))
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Hugging Face API details
API_URL1 = "https://api-inference.huggingface.co/models/JexCaber/TransLingo"
API_URL2 = "https://api-inference.huggingface.co/models/JexCaber/TransLingo-Terms"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}
HEADERS2 = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN2')}"}

def split_text_into_chunks(text, chunk_size=20, overlap=5):
    """Splits text into overlapping chunks to extract multiple terms."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def clean_term_output(output_text):
    """Cleans up extracted term output."""
    output_text = output_text.replace("Term: Term:", "Term:").replace("Term:", "", 1).strip()
    return output_text

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

        # Log response
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)

        response.raise_for_status()

        return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        print("Request Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/term-detection", methods=['POST'])
def term_detection():
    try:
        data = request.json
        text = data.get('text')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        chunks = split_text_into_chunks(text)
        extracted_terms = {}

        for chunk in chunks:
            # Define generation parameters
            generation_params = {
                "inputs": chunk,
                "parameters": {
                    "max_length": 150
                }
            }

            # Send request to Hugging Face API
            response = requests.post(API_URL2, headers=HEADERS2, json=generation_params)

            # Log response
            print("Status Code:", response.status_code)
            print("Response Text:", response.text)

            response.raise_for_status()
            response_json = response.json()

            # If response is a list, extract the first dictionary
            if isinstance(response_json, list) and len(response_json) > 0:
                response_json = response_json[0]  

            output_text = response_json.get('generated_text', "")

            output_text = clean_term_output(output_text)
            term, definition = output_text.split("| Definition: ", 1) if "| Definition: " in output_text else (output_text, "")

            term = term.strip().lower()  # Normalize

            if term and term not in extracted_terms:  # Avoid duplicates
                extracted_terms[term] = definition.strip()

        return jsonify({"extracted_terms": extracted_terms})

    except requests.exceptions.RequestException as e:
        print("Request Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Welcome to the TransLingo API!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)