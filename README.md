# 🏦 Bank Marketing Prediction

An end-to-end, beginner-friendly Machine Learning web application built using **Python**, **Scikit-Learn**, and **Streamlit**. The application predicts whether a bank customer is likely to subscribe to a term deposit based on demographic, financial, and marketing campaign features.

---

## 📌 1. Project Description
In retail banking, direct marketing campaigns (such as telemarketing phone calls) are frequently conducted to offer products like term deposits. However, contacting all customers indiscriminately results in high operational costs and customer annoyance. 

This project trains a **Logistic Regression** classification model using historical campaign data to identify potential subscribers in advance, helping banks focus their marketing efforts on high-probability leads.

---

## 🎯 2. Project Objective
- Analyze customer demographic attributes, credit history, and campaign interaction details.
- Clean and preprocess numerical and categorical variables using Scikit-Learn pipelines.
- Train an interpretable Machine Learning classification model (**Logistic Regression**).
- Deliver an interactive, clean **Streamlit web application** for real-time predictions and performance visualization.

---

## ✨ 3. Key Features
- **Interactive UI**: Clean, responsive web interface built with pure Streamlit.
- **Sidebar Customer Form**: Intuitive input controls (sliders, number inputs, select boxes) populated with real dataset values.
- **Instant Inference**: Computes subscription predictions (`yes` / `no`) alongside percentage confidence scores.
- **Model Evaluation Dashboard**: Shows test accuracy, precision, recall, F1-score, and a Seaborn confusion matrix heatmap.
- **Dataset Insights**: Shows dataset size, target class distribution, and sample data records.
- **Robust Error Handling**: Gracefully handles missing model files, absent datasets, and invalid inputs with friendly user alerts.

---

## 💻 4. Technologies Used
- **Python 3.10+**
- **Streamlit**: Web application framework.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-Learn**: Data preprocessing (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `Pipeline`) and `LogisticRegression`.
- **Matplotlib & Seaborn**: Data visualization and confusion matrix heatmap.
- **Joblib**: Model serialization and persistence.

---

## 🧠 5. Machine Learning Algorithm & Preprocessing
- **Algorithm**: `LogisticRegression(max_iter=1000, random_state=42)`
- **Data Splitting**: 80% training, 20% testing with **stratification** to preserve target class proportions.
- **Feature Engineering Pipeline**:
  - **Numerical Features** (`StandardScaler`): `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`.
  - **Categorical Features** (`OneHotEncoder(handle_unknown='ignore')`): `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`.
- **Target Variable (`y`)**:
  - `yes`: Customer subscribed to a term deposit.
  - `no`: Customer did not subscribe.

---

## 📂 6. Dataset
The project uses the standard **Bank Marketing Dataset** from the UCI Machine Learning Repository.
- **Total Records**: 4,521 customer records.
- **Total Columns**: 17 (16 features + 1 target variable `y`).
- **File Location**: `data/bank.csv`

---

## 📁 7. Project Structure
```text
bank-marketing-prediction/
│
├── app.py                  # Streamlit web application script
├── train_model.py          # Model training, evaluation & serialization script
├── requirements.txt        # Required Python packages
├── README.md               # Complete project documentation
│
├── data/
│   └── bank.csv            # Bank Marketing dataset (semicolon/comma separated)
│
└── models/
    ├── bank_model.pkl      # Complete end-to-end Pipeline (preprocessor + model)
    ├── preprocessor.pkl    # Standalone ColumnTransformer preprocessor
    └── model_metrics.json  # Cached test set evaluation metrics & summary stats
```

---

## 🚀 8. Installation & Setup

### Step 1: Clone or Navigate to the Project Directory
```bash
cd "Bank Marketing Prediction"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ 9. How to Train the Model
Run `train_model.py` to inspect the data, train the Logistic Regression pipeline, evaluate performance on the test set, and save the serialized model:

```bash
python train_model.py
```

### Expected Terminal Output:
```text
============================================================
 BANK MARKETING PREDICTION - MODEL TRAINING
============================================================

Loading dataset from: data\bank.csv

============================================================
 1. DATASET EXPLORATION & SUMMARY
============================================================
Dataset Shape: 4521 rows, 17 columns
...
============================================================
 5. MODEL EVALUATION ON TEST SET (20%)
============================================================

Accuracy : 89.28%
Precision: 56.36% (positive class: 'yes')
Recall   : 29.81% (positive class: 'yes')
F1-Score : 38.99%

Confusion Matrix [labels: ['no', 'yes']]:
               Predicted 'no'   Predicted 'yes'
Actual 'no' :      777            24
Actual 'yes':      73             31

============================================================
 6. SAVING TRAINED MODEL ARTIFACTS
============================================================
Saved full pipeline (preprocessor + model) to: models\bank_model.pkl
Saved standalone preprocessor to: models\preprocessor.pkl
Saved performance metrics summary to: models\model_metrics.json
```

---

## 🌐 10. How to Run the Streamlit Application
Launch the interactive web interface:

```bash
streamlit run app.py
```
After running this command, Streamlit will open the application in your default web browser (typically at `http://localhost:8501`).

---

## 🔍 11. Example Prediction

### Example Input:
- **Age**: 35
- **Job**: Management
- **Marital Status**: Single
- **Education**: Tertiary
- **Credit in Default**: No
- **Average Balance**: €5,000
- **Housing Loan**: No
- **Personal Loan**: No
- **Contact Type**: Cellular
- **Month**: April
- **Day**: 16
- **Contact Duration**: 1,200 seconds (~20 minutes)
- **Campaign Contacts**: 1
- **Pdays**: 180 days
- **Previous Contacts**: 3
- **Previous Outcome**: Success

### Expected Prediction Output:
```text
Prediction Result:
✅ Customer is likely to subscribe to the term deposit.

Prediction Probability:
- Likelihood to Subscribe ('Yes'): 99.27%
- Likelihood NOT to Subscribe ('No'): 0.73%
```

---

## 📊 12. Interpreting the Prediction
- **Probability Threshold**: The model calculates the mathematical probability of subscription using the sigmoid function in Logistic Regression.
- **Actionable Insight**:
  - **High Probability (`> 50%`)**: The customer demonstrates strong engagement (e.g., longer call durations, successful previous campaign response, healthy savings balance). The sales team should prioritize following up with this lead.
  - **Low Probability (`< 50%`)**: The customer has low affinity for term deposits based on recent or past contacts. Avoiding repeated cold calls saves bank resources and prevents negative brand sentiment.

---

## 👥 Author & License
- **Project**: College / Student Machine Learning Demonstration
- **License**: MIT License
