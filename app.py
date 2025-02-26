from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app= Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/JexCaber/TransLingo"

HEADERS ={"Authorization": f"Bearer {os.getenv('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')}"}

@app.route("/simplify-text", methods=['POST'])
def simplify_text():
    try:
        data = request.json
        text =data.get('text')

        if not text:
            return({"error": "No Text Provided"}), 400
        
        response = requests.post(API_URL, header = HEADERS, json={"inputs":text})
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exception.RequestException as e:
        return jsonify({"error":str(e)}), 500
    

def home():
    return "Welcome"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)