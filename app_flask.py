# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 23:41:48 2025

@author: ramib
"""

from flask import Flask, request, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Model and Tokenizer Paths
model_name = "NbAiLab/nb-bert-base"  # Change to your specific model if needed
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_nb_bert")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_nb_bert")

# Home route to display the input form
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Predict route to handle text input and make predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the input text from the form
        data = request.form["text"]
        
        # Tokenize and predict
        inputs = tokenizer(data, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
        
        # Map prediction to label
        label = "Toxic" if prediction == 1 else "Non-Toxic"
        return render_template("index.html", prediction=label, text=data)

    except Exception as e:
        # Handle any errors and display on the page
        return render_template("index.html", prediction=f"Error: {str(e)}", text=data)

# Run the app
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
