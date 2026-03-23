# AI_ML-Internship-Tasks-phase-2
## Overview
This repository contains **three AI/ML projects** completed during the DevelopersHub AI/ML Engineering Internship.  

**Included Projects:**
1. News Topic Classification using BERT  
2. Customer Churn Prediction Pipeline  
3. Auto Tagging Support Tickets using LLM  

Each project demonstrates practical AI/ML applications for real-world tasks.

---

## Table of Contents
- [Project 1: News Topic Classification](#project-1-news-topic-classification)
- [Project 2: Customer Churn Prediction](#project-2-customer-churn-prediction)
- [Project 3: Auto Tagging Support Tickets](#project-3-auto-tagging-support-tickets)
- [How to Run](#how-to-run)
- [Author & Contact](#author--contact)
- [Acknowledgments](#acknowledgments)

---

## Project 1: News Topic Classification
<details>
<summary>Click to expand</summary>

**Goal:** Classify news headlines into 4 categories: World, Sports, Business, Sci/Tech.

**Dataset:** AG News (subset for CPU: 5,000 train, 1,000 test)  

**Tools:** Python, PyTorch, Hugging Face Transformers & Datasets, pandas, numpy, scikit-learn, matplotlib, seaborn  

**Model:** BERT (`bert-base-uncased`) with classification head  
- 3 training epochs  
- Batch size 16  
- Learning rate 5e-5  

**Results:**  
- Accuracy: 91.2%  
- Weighted F1-score: 0.9115  

**Sample Prediction:**  
- "Lakers defeat Warriors in overtime thriller" → Sports (98% confident)  
- "Tesla stock surges after earnings report" → Business (94% confident)  

**Folder Structure:**

news_classifier_bert/
├── news_classifier_bert.ipynb
├── news_classifier_bert_model/
├── results/
├── logs/
└── README.md


</details>

---

## Project 2: Customer Churn Prediction
<details>
<summary>Click to expand</summary>

**Goal:** Predict which customers will leave (churn) using Telco dataset (~7,000 customers).

**Tools:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib  

**Steps:**  
1. Clean data (remove customerID, fix missing values, encode target)  
2. Scale numerical features, one-hot encode categorical features  
3. Build ML pipelines for preprocessing + model  
4. Train Logistic Regression and Random Forest (with tuning)  
5. Evaluate using Accuracy, Precision, Recall, F1-score  

**Results:**  
| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|---------|----------|-------|----------|
| Logistic Regression   | 0.80    | 0.66     | 0.55  | 0.60     |
| Random Forest (Tuned) | 0.79    | 0.65     | 0.51  | 0.57     |

**Best Model:** Random Forest (`n_estimators=100`, `max_depth=20`, `min_samples_split=5`)  

**Folder Structure:**

churn_prediction_pipeline/
├── churn_prediction_pipeline.ipynb
├── churn_prediction_pipeline.pkl
├── telco_churn.csv
└── README.md


</details>

---

## Project 3: Auto Tagging Support Tickets
<details>
<summary>Click to expand</summary>

**Goal:** Automatically tag support tickets into categories using Large Language Models (LLM).  

**Dataset:** Synthetic tickets (50 examples), categories: Billing, Technical, Account, Feature Request, General Inquiry  

**Tools:** Python, Hugging Face Transformers, PyTorch, pandas, scikit-learn  

**Model:** `facebook/bart-large-mnli` (zero-shot, no training)  

**Steps:**  
1. Use zero-shot classification for automatic tagging  
2. Optionally, few-shot learning by giving a few examples to improve accuracy  
3. Get top 3 predictions with confidence scores  

**Results:**  
- Accuracy: 58%  
- Average confidence: 38%  
- High-confidence tickets (>70%): 5/50  

**Sample Prediction:**  
Ticket: "I was charged twice for my subscription"  
- Top 3: Billing (95%), Refund Request (3%), Subscription (1%)  

**Folder Structure:**

support_ticket_tagging/
├── support_ticket_tagging.ipynb
├── support_ticket_predictions.csv
└── README.md


</details>

---

## How to Run

**Install Dependencies:**
```bash
pip install transformers datasets torch pandas numpy scikit-learn matplotlib seaborn joblib jupyter accelerate

Run each project:

# Project 1: BERT News Classification
jupyter notebook news_classifier_bert.ipynb

# Project 2: Churn Prediction Pipeline
jupyter notebook churn_prediction_pipeline.ipynb

# Project 3: LLM Ticket Auto-Tagging
jupyter notebook support_ticket_tagging.ipynb
Author
Somia Iqbal
AI/ML Engineering Intern, DevelopersHub Corporation
GitHub: [https://github.com/somiaiqbal/AI_ML-Internship-Tasks-phase-2]
