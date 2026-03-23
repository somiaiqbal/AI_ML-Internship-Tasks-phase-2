# Customer Churn Prediction - ML Pipeline

## Objective
Build a production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset.

## Dataset
- **Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Total Samples:** ~7,000 customers  
- **Features:** 19 features including demographics, services, and account information  
- **Target Variable:** Churn (Yes/No)  

## Technologies Used
- Python 3.x  
- pandas - Data manipulation  
- numpy - Numerical operations  
- scikit-learn - Machine learning pipeline and models  
- matplotlib & seaborn - Data visualization  
- joblib - Model serialization  

## Methodology

### 1. Data Preprocessing
- Removed `customerID` column (non-predictive)  
- Converted `TotalCharges` to numeric  
- Handled missing values using median imputation  
- Encoded target variable (`Churn`: Yes → 1, No → 0)  

### 2. Feature Engineering
- **Numerical Features:** `tenure`, `MonthlyCharges`, `TotalCharges`  
- **Categorical Features:** `gender`, `Partner`, `Dependents`, `PhoneService`, etc.  
- Applied `StandardScaler` to numerical features  
- Applied `OneHotEncoder` to categorical features  

### 3. Pipeline Construction
- **Preprocessing Pipeline:** Handles imputation, scaling, and encoding  
- **Model Pipeline:** Combines preprocessing with classifier  
- **Full Pipeline:** End-to-end workflow from data to predictions  

### 4. Models Trained
1. **Logistic Regression**  
   - Baseline model  
   - Fast training and prediction  
2. **Random Forest Classifier**  
   - More complex model  
   - Hyperparameter tuning using `GridSearchCV`  
   - Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`  

### 5. Model Evaluation
- Metrics used:  
  - **Accuracy:** Overall correctness  
  - **Precision:** Correctness of predicted churns  
  - **Recall:** How many actual churns were caught  
  - **F1-Score:** Harmonic mean of precision and recall  

#### Model Performance
| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|---------|----------|-------|----------|
| Logistic Regression   | 0.80    | 0.66     | 0.55  | 0.60     |
| Random Forest (Tuned) | 0.79    | 0.65     | 0.51  | 0.57     |

### Best Model
Random Forest with optimized hyperparameters:  
- `n_estimators`: 100  
- `max_depth`: 20  
- `min_samples_split`: 5  

### Key Insights
1. `Tenure` and `MonthlyCharges` are strong predictors of churn  
2. Class imbalance: ~73% non-churn, ~27% churn  
3. Random Forest has higher precision but lower recall than Logistic Regression  
4. Full pipeline is production-ready and deployable using `joblib`  

## Project Structure

churn_prediction_pipeline/
│
├── churn_prediction_pipeline.ipynb # Main notebook with code
├── churn_prediction_pipeline.pkl # Saved pipeline model
├── telco_churn.csv # Dataset
└── README.md # Project documentation


## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
Steps
Download the Telco Churn dataset from Kaggle
Place telco_churn.csv in the project folder
Open Jupyter Notebook:
jupyter notebook
Run all cells in churn_prediction_pipeline.ipynb
Using the Saved Pipeline
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('churn_prediction_pipeline.pkl')

# Make predictions on new data
new_data = pd.DataFrame([...])  # Your new customer data
predictions = pipeline.predict(new_data)
Skills Demonstrated
Data preprocessing and cleaning
Feature engineering (numerical + categorical)
Pipeline construction with scikit-learn
Hyperparameter tuning using GridSearchCV
Model evaluation using multiple metrics
Model serialization for production deployment
Learning Outcomes
Built reproducible ML pipelines
Handled mixed data types (numerical + categorical)
Gained experience with hyperparameter tuning
Practiced model evaluation and comparison
Developed a production-ready ML workflow
Author
Somia Iqbal
DevelopersHub Corporation - AI/ML Engineering Intern
Contact
GitHub: [https://github.com/somiaiqbal/AI_ML-Internship-Tasks-phase-2]
This project is part of the DevelopersHub Corporation AI/ML Engineering Internship Program.
