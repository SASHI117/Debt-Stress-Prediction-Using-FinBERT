# 💳 Debt Stress Prediction using FinBERT  
### Financial Stress Classification Model (Low / Medium / High Stress)

This project implements an **AI-powered Financial Stress Detection System** using a fine-tuned **FinBERT** model.  
It analyzes financial messages such as SMS alerts, EMI reminders, credit card notices, and banking updates,  
and classifies them into:

- **Low Stress (0)**
- **Medium Stress (1)**
- **High Stress (2)**

This model is highly relevant for **fintech, credit risk analysis, collections, financial well-being, and  
customer hardship identification**, especially for companies like **CredResolve**.

---

## 🚀 Project Highlights

✔ Fine-tuned **FinBERT** for stress classification  
✔ **2100 synthetic training samples** (balanced dataset)  
✔ **NVIDIA T4 GPU** used for training  
✔ Achieved **100% evaluation accuracy** on validation set  
✔ Supports **batch prediction** + probability scores  
✔ Clean, reproducible notebook for end-to-end pipeline  

---

## 📌 Features

- Fully automated **synthetic dataset generation**
- Text preprocessing + tokenization using FinBERT
- Transfer learning with **HuggingFace Trainer**
- Cross-entropy loss optimization
- Evaluation using **Accuracy** and **F1-score**
- Inference function for real-world financial messages
- Exportable and deployable trained model

---

## 📂 Project Structure

```

DebtStressPrediction/
│
├── dataset/                  # Synthetic dataset (2100 samples)
├── finbert_stress_model/     # Saved fine-tuned model + tokenizer
├── DebtStressPrediction.ipynb  # Main Colab notebook
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies

```

---

## 🧠 Model Architecture

### 🔹 Base Model  
**FinBERT (BERT-base, financial domain-specific)**  
Pretrained on:  
- Earnings Calls  
- Financial Reports  
- Analyst Statements  
- Market Sentiment Data  

### 🔹 Fine-Tuning Details  
- Loss: Cross Entropy  
- Optimizer: AdamW  
- LR: 2e-5  
- Epochs: 3  
- Batch Size: 8  
- Max Length: 128  

---

## 📊 Dataset

A balanced synthetic dataset was created due to deprecation of public finance datasets on HuggingFace.

| Stress Level | Count |
|--------------|-------|
| Low Stress   | 700   |
| Medium Stress| 700   |
| High Stress  | 700   |
| **Total**    | **2100** |

### Examples:

**High Stress:**  
- “Your EMI payment is overdue by 25 days.”  
- “Your credit card bill of $3500 is long overdue.”

**Medium Stress:**  
- “Your account balance is below minimum requirement.”  
- “Your bill of $1200 is due tomorrow.”

**Low Stress:**  
- “Thank you! No dues remaining on your account.”  
- “Your credit score has improved this month.”

---

## 🏋️ Training Results

The model achieves:

- **Accuracy:** 100%  
- **F1-score:** 100%  
- **Validation Loss:** ~0.00046  

This is expected due to:
- Balanced dataset  
- Template-based synthetic messages  
- FinBERT’s strong understanding of financial language  

---

## 🖼️ Results Screenshot

---

<img width="1919" height="764" alt="Screenshot 2025-12-10 002630" src="https://github.com/user-attachments/assets/3af339f6-0aac-47c3-8180-4d43fa5172cc" />

---

## 🧪 Example Predictions

**Input:**  
`"Your EMI payment is overdue by 15 days."`  
**Output:**  
`HIGH STRESS`  

**Input:**  
`"Thank you! No dues remaining on your account."`  
**Output:**  
`LOW STRESS`  

**Input:**  
`"Your account balance is below the minimum requirement."`  
**Output:**  
`MEDIUM STRESS`  

---

## 🛠️ Installation

```bash
pip install torch transformers datasets
````

Clone the repo:

```bash
git clone https://github.com/your-username/DebtStressPrediction.git
cd DebtStressPrediction
```

---

## ▶️ Usage (Inference)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("finbert_stress_model")
tokenizer = AutoTokenizer.from_pretrained("finbert_stress_model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs).item()
    mapping = {0: "LOW STRESS", 1: "MEDIUM STRESS", 2: "HIGH STRESS"}
    return mapping[label], probs
```

---

## 📈 Applications

### For **CredResolve**

* Customer stress detection
* Prioritization of high-stress accounts
* Enhanced risk communication
* Automated hardship support workflows

### Banking & FinTech

* Credit underwriting
* Behavioral risk scoring
* Early delinquency detection
* Personal finance coaching tools

---

## 🔮 Future Improvements

* Train on **real customer datasets**
* Expand stress scale (0–5 instead of 0–3)
* Add multilingual financial stress detection
* Deploy as REST API or Streamlit web app

---

## 🤝 Contributions

Pull requests are welcome!
If you'd like new features or improvements, feel free to open an issue.

---

## 📜 License

This project is licensed under the MIT License.

---

## ❤️ Author

**Sashi Vardhan Pragada**
B.Tech ECE (AI/ML) — GITAM University

---
