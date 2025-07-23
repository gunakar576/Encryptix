# Encryptix

# Internship Projects at Encryptix 

Welcome to my internship portfolio! During my time at Encryptix , I worked on three end-to-end machine learning projects, each targeting a real-world classification problem.

This repository includes:
- Exploratory Data Analysis (EDA)
-  Feature Engineering
-  Model Building with Scikit-learn
- Handling Class Imbalance
- Evaluation Metrics

---

##  Project 1: Customer Churn Prediction

###  Objective
Predict whether a bank customer will churn using a Random Forest Classifier.

###  Dataset
- `Churn_Modelling.csv`

###  Tools & Libraries
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn (RandomForestClassifier, StandardScaler, metrics)

###  Highlights
- Cleaned dataset and selected relevant features
- Encoded categorical columns like `Gender` and `Geography`
- Applied feature scaling
- Trained a Random Forest model with `class_weight='balanced'`
- Evaluated using classification metrics

###  Results
- **Accuracy**: 86.15%  
- **Precision**: 78.76%  
- **Recall**: 43.73%  
- **F1 Score**: 56.24%

---

##  Project 2: Spam SMS Detection using SVM

### Objective
Classify SMS messages as spam or ham using NLP techniques and Support Vector Machine.

###  Dataset
- `spam.csv`

### Tools & Libraries
- Python, Pandas  
- Seaborn, Matplotlib  
- Scikit-learn (`TfidfVectorizer`, `SVC`, `train_test_split`, `classification_report`)

###  Highlights
- Cleaned raw dataset and retained only relevant columns
- Mapped `ham → 0`, `spam → 1`
- Created `Message_Length` feature
- Visualized class distribution and message lengths
- Converted text using `TfidfVectorizer`
- Trained an SVM classifier with a linear kernel

###  Results
- **Accuracy**: 97.93%  
- **Confusion Matrix**:
- **Classification Report**:
- Precision (spam): 97%
- Recall (spam): 87%
- F1 Score (spam): 92%

---

## 💳 Project 3: Credit Card Fraud Detection

### 📌 Objective
Detect fraudulent credit card transactions in a highly imbalanced dataset.

### 📁 Dataset
- `creditcard.csv` (anonymized)

### 🛠️ Tools & Libraries
- Python, Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn (`RandomForestClassifier`, `LogisticRegression`, `SMOTE`, metrics)

### 🔍 Highlights
- Analyzed imbalanced class distribution
- Used **SMOTE** for oversampling the minority class or used `class_weight='balanced'`
- Trained both Logistic Regression and Random Forest models
- Focused on **recall** to reduce false negatives
- Evaluated performance using confusion matrix, precision, recall, and F1-score

### ✅ Results
- Improved fraud detection after balancing
- Emphasis on high recall for fraud class
- Demonstrated end-to-end fraud detection workflow

---

## 📁 Folder Structure

```bash
Encryptix-Internship/
│
├── 📁 Project_1_Customer_Churn/
│   └── Churn_Prediction.ipynb
│
├── 📁 Project_2_Spam_SMS_Detection/
│   └── SPAM_SMS_DETECTION.ipynb
│
├── 📁 Project_3_Credit_Card_Fraud/
│   └── CREDIT_CARD_FRAUD_DETECTION.ipynb
│
└── README.md

