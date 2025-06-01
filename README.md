
🛡️ Hate Speech Detection using BERT

This project builds a hate speech detection model using a fine-tuned Norwegian BERT model (NbAiLab/nb-bert-large). The model classifies online comments as toxic or non-toxic using the multilingual_textdetox dataset.

---

Overview

- Dataset: textdetox/multilingual_toxicity_dataset (English subset)
- Model: NbAiLab/nb-bert-large, fine-tuned and hosted on Hugging Face
- Task: Binary classification (toxic vs. non-toxic)
- Includes:
  - Tokenization with Hugging Face Transformers
  - Training using the Trainer API
  - Evaluation metrics (accuracy, precision, recall, F1)

---

Features

- Fine-tuning BERT for binary text classification
- Integration with Hugging Face Datasets and Transformers
- Export of trained model and tokenizer
- Web interface with Flask (`app_flask.py`) for user input and prediction
- Hosted model for download and reuse

---

🔗 Pretrained Model

The fine-tuned model is publicly available here:  
👉 https://huggingface.co/RamiBadleh/Heat_speach_detection

---

Tech Stack

- Python
- Hugging Face Transformers & Datasets
- PyTorch
- Scikit-learn
- Flask (for web interface)

---

How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Train the model (optional):
   Run the notebook: `heat speach detection.ipynb`

3. Launch the web app:
   python app_flask.py

---

License

This project is licensed under the MIT License.

---

Acknowledgments

- Dataset: Hugging Face `textdetox`
- Model: `NbAiLab/nb-bert-large`
- Hosted at: `RamiBadleh/Heat_speach_detection` on Hugging Face
- Developed as a school project to explore toxic content detection
