#  Customer Churn Analysis & Prediction
> An **end-to-end customer retention analytics system** — from deep exploratory data analysis identifying the root causes of churn, through a class-imbalance-aware classification model, to a **live Flask web application** that gives retention teams real-time churn risk scores for any customer.

---

##  Table of Contents

- [Project Overview]
- [Business Problem]
- [Architecture]
- [Tech Stack]
- [EDA Findings]
- [Modelling Approach]
- [Class Imbalance Strategy]
- [Evaluation Metrics]
- [Project Structure]
- [Getting Started]
- [Results]
- [Future Improvements]

---

##  Project Overview

Customer churn — when a customer cancels their subscription — is one of the most expensive events in a recurring revenue business. In telecom, acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a complete, production-ready churn prediction system that:

1. **Diagnoses** which customer characteristics are most strongly associated with churn
2. **Predicts** which customers are at risk, ranked by probability
3. **Deploys** a web tool that retention teams can use in their daily workflow — no data science knowledge required

---

##  Business Problem

**Dataset:** Telco Customer Churn dataset (~7,000 customers, 21 features)  
**Target variable:** `Churn` — binary (Yes/No)  
**Class distribution:** ~26% churn, ~74% retain (imbalanced)

**Cost asymmetry:**
- **False Negative** (predict "stay", customer churns): Lose full future lifetime value — potentially $500–2,000 per customer
- **False Positive** (predict "churn", customer stays): Spend a retention incentive unnecessarily — typically $20–50

This 10–40× cost asymmetry drives every modelling decision in this project.

---

## 🏗️ Architecture

```
Raw Telco Data (CSV)
		│
		▼
┌───────────────────┐
│  Exploratory      │  ← Churn rates by segment, correlation analysis,
│  Data Analysis    │    distribution plots, class imbalance assessment
└────────┬──────────┘
		 │
		 ▼
┌───────────────────┐
│  Preprocessing    │  ← Encode categoricals, scale numerics,
│  & Feature Eng.   │    SMOTE oversampling (training data only)
└────────┬──────────┘
		 │
		 ▼
┌───────────────────┐
│  Model Training   │  ← Logistic Regression, Decision Tree,
│  & Comparison     │    Random Forest (winner), SVM
└────────┬──────────┘
		 │
		 ▼
┌───────────────────┐
│  Threshold Tuning │  ← PR curve analysis
│                   │    Optimal threshold: 0.35 (recall-maximising)
└────────┬──────────┘
		 │
		 ▼
┌───────────────────┐
│  Flask Web App    │  ← Customer form → churn probability
│  (app.py)         │    Live inference endpoint: localhost:5000
└───────────────────┘
```

---

##  Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **ML Framework** | scikit-learn | Classification models, pipelines, evaluation |
| **Imbalance Handling** | imbalanced-learn | SMOTE oversampling for minority class |
| **Web Deployment** | Flask | REST API + HTML form serving |
| **Data Processing** | Pandas, NumPy | Wrangling, feature engineering, aggregation |
| **Visualisation** | Matplotlib, Seaborn | EDA charts, confusion matrices, ROC/PR curves |
| **Serialisation** | Pickle | Model artifact persistence for deployment |
| **Frontend** | HTML/CSS | Customer input form (home.html) |
| **Language** | Python 3.9+ | Core development |

---

##  EDA Findings — Churn Drivers

### Key Insight 1: Contract Type is the Dominant Predictor
| Contract Type | Churn Rate |
|---|---|
| Month-to-month | **42.7%** |
| One year | 11.3% |
| Two year | **2.8%** |

**Action implication:** Proactively offer contract upgrades to month-to-month customers in their first 6 months.

### Key Insight 2: Tenure — The 12-Month Risk Window
Customers who survive past 12 months show dramatically lower churn probability. The first year is the critical retention window.

### Key Insight 3: The Fiber Optic Paradox
Fiber optic customers pay **35% more** on average than DSL customers but churn at **twice the rate** — indicating a quality/value dissatisfaction, not financial hardship. Retention strategy: service quality intervention, not discounts.

### Key Insight 4: Payment Method as an Engagement Proxy
| Payment Method | Churn Rate |
|---|---|
| Electronic check | **45.3%** |
| Mailed check | 19.1% |
| Bank transfer (auto) | 16.7% |
| Credit card (auto) | 15.2% |

Auto-pay correlates with commitment; electronic check users are the highest-risk segment.

### Key Insight 5: The Interaction Effect
Month-to-month + high monthly charges + no online security + electronic check = **68% predicted churn probability** — the highest-risk customer profile identified.

---

##  Modelling Approach

### Models Trained & Compared

| Model | Recall | Precision | ROC-AUC | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.74 | 0.62 | 0.83 | 0.67 |
| Decision Tree | 0.71 | 0.58 | 0.78 | 0.64 |
| **Random Forest** | **0.78** | **0.67** | **0.86** | **0.72** |
| SVM (RBF) | 0.75 | 0.65 | 0.84 | 0.70 |

*All results at tuned threshold (0.35), evaluated on 20% held-out test set.*

**Winner: Random Forest** — best recall (fewest missed churners) with competitive precision.

---

##  Class Imbalance Strategy

The dataset has ~26% churn — a naive model predicting "no churn" for everyone achieves 74% accuracy but catches zero churners. Three strategies applied:

### 1. SMOTE (Synthetic Minority Over-sampling)
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline([
	('preprocessor', preprocessor),
	('smote', SMOTE(random_state=42)),  # Applied ONLY inside training folds
	('classifier', RandomForestClassifier(class_weight='balanced'))
])
```
 **SMOTE is applied exclusively inside cross-validation training folds** — applying it before splitting would constitute data leakage.

### 2. Class Weighting
```python
RandomForestClassifier(class_weight='balanced')
# Automatically weights minority class inversely proportional to frequency
```

### 3. Decision Threshold Tuning
```python
# Default threshold: 0.5 (assumes symmetric costs)
# Tuned threshold: 0.35 (maximises recall given cost asymmetry)
# Selected by: minimising false negative rate on PR curve
```

---

## 📏 Evaluation Metrics — Why Recall, Not Accuracy

```
Accuracy = (TP + TN) / All   ← MISLEADING with imbalanced classes
Recall   = TP / (TP + FN)    ← "What fraction of churners did we catch?" ✅
Precision = TP / (TP + FP)   ← "What fraction of our alerts were real?"
ROC-AUC                      ← Ranking ability across all thresholds ✅
PR-AUC                       ← More informative than ROC under imbalance ✅
```

**Business framing:** At a retention call cost of $30 and average revenue-at-risk of $600, a precision of 30% still yields a positive ROI: 10 calls × $30 = $300 cost vs 3 saves × $600 = $1,800 revenue (6× ROI).

---

##  Flask Web Application

The trained model is served via a Flask web application — making predictions accessible to non-technical retention teams.

### How It Works

```python
# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask("__name__")
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def loadPage():
	return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
	# Extract form inputs
	input_data = {
		'gender': request.form['query1'],
		'SeniorCitizen': int(request.form['query2']),
		'tenure': int(request.form['query3']),
		'MonthlyCharges': float(request.form['query4']),
		'Contract': request.form['query5'],
		# ... additional fields
	}
	df = pd.DataFrame([input_data])
	prediction = model.predict_proba(df)[0][1]
	churn_risk = f"{prediction:.0%}"
	return render_template('home.html', output1=churn_risk, ...)

if __name__ == "__main__":
	app.run()
```

### Running the App

```bash
# Start the Flask server
python app.py

# Open in browser
# http://127.0.0.1:5000
```

The interface shows a customer form. A retention analyst enters the customer's details and receives:
> *"This customer has a **73% churn risk**. Recommended action: Priority retention call."*

---

##  Project Structure

```
churn_analysis/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Raw Telco dataset
│
├── notebooks/
│   ├── Churn Analysis - EDA.ipynb               # 🔍 Exploratory analysis
│   └── Churn Analysis - Model Building.ipynb    # 🤖 Model training & evaluation
│
├── src/
│   ├── preprocessing.py                         # Encoding, scaling utilities
│   └── model_utils.py                           # Training & evaluation helpers
│
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── references/                                  # Relevant papers & data dictionary
│
├── app.py                                       # 🌐 Flask web application
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting Started

```bash
# 1. Clone repository
git clone https://github.com/Georginh0/churn_analysis.git
cd churn_analysis

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run EDA notebook
jupyter notebook notebooks/Churn\ Analysis\ -\ EDA.ipynb

# 5. Train the model
jupyter notebook notebooks/Churn\ Analysis\ -\ Model\ Building.ipynb

# 6. Launch the web app
python app.py
# → Open http://127.0.0.1:5000
```

---

##  Results

| Metric | Value | Business Meaning |
|---|---|---|
| **Recall** | **0.78** | 78% of churners identified before they leave |
| **Precision** | 0.67 | 67% of flagged customers genuinely at risk |
| **ROC-AUC** | 0.86 | Strong ranking ability across risk tiers |
| **F1-Score** | 0.72 | Balanced metric at tuned threshold |

At the company's estimated retention economics, the model generates an estimated **$4.2 revenue for every $1 spent** on retention actions triggered by its predictions.

---

## Future Improvements

- [ ] **Drift monitoring** — PSI-based alerts when customer feature distributions shift
- [ ] **SHAP explainability** — Per-customer feature attribution ("this customer is high-risk because of X")
- [ ] **Risk tiering** — Three-bucket system (High/Medium/Low) with different retention playbooks per tier
- [ ] **CRM integration** — Direct webhook to Salesforce/HubSpot to create retention tasks automatically
- [ ] **Lifelong learning** — Online model updates as new churn/retain ground truth flows in
- [ ] **Dockerise** — Container deployment for production-safe multi-worker serving

---

##  License

This project is licensed under the MIT License — see [LICENCE](./LICENCE).

---

## 👤 Author

**George Dogo** — Data Scientist  
📧 George_dogo@aol.com | 🐙 [github.com/Georginh0](https://github.com/Georginh0)

*If this project helped you understand churn modelling, please ⭐ the repo!*
