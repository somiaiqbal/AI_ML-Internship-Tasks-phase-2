# News Topic Classifier Using BERT

## Objective
Fine-tune a BERT transformer model to classify news headlines into topic categories using the AG News dataset.

## Dataset
- **Name:** AG News Dataset  
- **Source:** Hugging Face Datasets  
- **Size:** 120,000 training samples, 7,600 test samples  
- **Categories:**  
  1. World  
  2. Sports  
  3. Business  
  4. Sci/Tech  

> For faster CPU training, a subset of 5,000 training and 1,000 test samples was used.

## Technologies Used
- Python 3.x  
- PyTorch - Deep learning framework  
- Hugging Face Transformers - BERT model and tokenizer  
- Hugging Face Datasets - AG News dataset  
- scikit-learn - Evaluation metrics  
- pandas, numpy - Data manipulation  
- matplotlib, seaborn - Visualization  

## Model Architecture
- **Base Model:** `bert-base-uncased`  
- **Type:** BERT (Bidirectional Encoder Representations from Transformers)  
- **Parameters:** ~110 million  
- **Pre-training:** BookCorpus + English Wikipedia  
- **Fine-tuning:** Added classification head for 4 categories  

## Methodology

### 1. Data Loading
- Loaded AG News dataset from Hugging Face  
- Created smaller subset (5k train, 1k test)  
- Checked class distribution (balanced dataset)  

### 2. Text Preprocessing
- Tokenized text using `BertTokenizer`  
- Max sequence length: 128 tokens  
- Applied padding and truncation  
- Created attention masks  

### 3. Model Fine-Tuning
- Added classification layer for 4 classes  
- Training configuration:  
  - Epochs: 3  
  - Batch Size: 16 (train), 32 (eval)  
  - Learning Rate: 5e-5  
  - Warmup Steps: 500  
  - Weight Decay: 0.01  

### 4. Training Process
- Used Hugging Face `Trainer` API  
- Evaluated after each epoch  
- Saved best model based on F1 score  
- Training time: ~12 minutes on CPU  

### 5. Evaluation
- Metrics: Accuracy, Weighted F1 score  
- Generated classification report & confusion matrix  
- Tested on custom news headlines  

#### Performance Metrics
| Metric        | Score  |
|---------------|-------|
| Accuracy      | 91.2% |
| Weighted F1   | 0.9115|
| Training Loss | 0.2847|

#### Per-Class Performance
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|--------|
| World    | 0.88      | 0.91   | 0.89     | 250    |
| Sports   | 0.97      | 0.95   | 0.96     | 250    |
| Business | 0.89      | 0.88   | 0.89     | 250    |
| Sci/Tech | 0.91      | 0.90   | 0.90     | 250    |

## Key Insights
- **Transfer Learning:** Pre-trained BERT achieves 91%+ accuracy with minimal fine-tuning  
- **Category Patterns:** Sports easiest to classify; World and Business sometimes overlap  
- **Model Behavior:** High-confidence predictions are mostly correct  
- **Training Efficiency:** 3 epochs sufficient for convergence  

## Sample Predictions
| Headline                                         | Predicted       | Actual    |
|-------------------------------------------------|----------------|-----------|
| Biden announces new economic policy for 2024    | World (89.5%)  | World     |
| Lakers defeat Warriors in overtime thriller     | Sports (98.3%) | Sports    |
| Tesla stock surges after earnings report        | Business (94.7%)| Business |
| Scientists discover new planet in distant solar system | Sci/Tech (98.2%) | Sci/Tech |

## Project Structure

news_classifier_bert/
│
├── news_classifier_bert.ipynb # Main training notebook
├── news_classifier_bert_model/ # Saved model files
│ ├── config.json
│ ├── pytorch_model.bin
│ └── tokenizer files
├── results/ # Training checkpoints
├── logs/ # Training logs
└── README.md # Project documentation


## How to Run

### Prerequisites
```bash
pip install transformers datasets torch pandas numpy scikit-learn matplotlib seaborn jupyter accelerate
Training
Open Jupyter Notebook
Run all cells in news_classifier_bert.ipynb
Training takes ~10-15 minutes on CPU
Model automatically saves to ./news_classifier_bert_model/
Inference (Using Saved Model)
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model
model = BertForSequenceClassification.from_pretrained('./news_classifier_bert_model')
tokenizer = BertTokenizer.from_pretrained('./news_classifier_bert_model')

# Predict
text = "Your news headline here"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()

labels = ['World', 'Sports', 'Business', 'Sci/Tech']
print(f"Category: {labels[prediction]}")
Skills Demonstrated
NLP with Transformers (BERT)
Transfer Learning & Fine-tuning
Multi-class Text Classification
Tokenization & Attention Masks
Model Evaluation (Accuracy, F1, Confusion Matrix)
Hugging Face Ecosystem (Transformers, Datasets, Trainer)
Model Deployment & Saving
Learning Outcomes
Understood transformer architecture and attention mechanism
Learned effective use of Hugging Face libraries
Practiced fine-tuning and tokenization
Gained hands-on experience in text classification
Real-World Applications
News Aggregation: Auto-categorize news articles
Content Recommendation: Suggest articles by topic
Email Filtering: Classify emails by topic
Document Organization: Automatic sorting
Social Media Monitoring: Track trending topics
Potential Improvements
Train on full 120k dataset for higher accuracy
Increase epochs & use early stopping
Hyperparameter tuning: learning rate, batch size
Explore BERT-large or RoBERTa
Data augmentation (back-translation, paraphrasing)
Multi-label classification for overlapping categories
Limitations
Subset dataset due to CPU constraints
English-only news
Limited categories
Model size (~440MB) may be large for mobile
Future Work
Deploy via Streamlit/Gradio
Real-time news classification API
Multi-language support
Fine-tune for domain-specific news
Author
Somia Iqbal
DevelopersHub Corporation - AI/ML Engineering Intern
Contact
GitHub: [https://github.com/somiaiqbal/AI_ML-Internship-Tasks-phase-2]
Acknowledgments
Hugging Face (Transformers library & AG News dataset)
Google Research (BERT model)
DevelopersHub Corporation (Internship support)

This project demonstrates practical application of state-of-the-art NLP techniques for real-world text classification tasks.
