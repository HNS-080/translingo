from flask import Flask, request, jsonify  # Import Flask, request, and jsonify
import requests
import os
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/JexCaber/TransLingo"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}

@app.route("/simplify-text", methods=['POST'])
def simplify_text():
    try:
        # Get JSON data from the request
        data = request.json
        text = data.get('text')

        # Check if text is provided
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Call the Hugging Face API
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})  # Fix: Use 'headers' instead of 'header'
        response.raise_for_status()  # Raise an error for bad responses

        # Return the Hugging Face API response
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:  # Fix: Use 'requests.exceptions' instead of 'requests.exception'
        # Handle request errors
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Welcome to the TransLingo API!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)