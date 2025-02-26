from flask import Flask, request, jsonify, make_response
from flask_cors import CORS  
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)
CORS(app)  

model_path = 'JexCaber/Translingo' 
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

@app.route("/simplify-text", methods=['POST'])
def simplify_text():
    input_text = request.json.get('text', '')

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150, top_k=50)
    simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(simplified_text)
    return make_response(jsonify({"simplifiedText": simplified_text}), 200)