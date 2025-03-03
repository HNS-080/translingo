from flask import Flask, request, jsonify  # Import Flask, request, and jsonify
import requests
import os
print("HUGGINGFACE_TOKEN:", os.getenv('HUGGINGFACE_TOKEN'))
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

        # Define generation parameters
        generation_params = {
            "inputs": text,
            "parameters": {
                "max_length": 150
            }
        }

        # Send request to Hugging Face API
        response = requests.post(API_URL2, headers=HEADERS, json=generation_params)

        # Log response
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)

        response.raise_for_status()

        return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        print("Request Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Welcome to the TransLingo API!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)