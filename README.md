
# 🏦 Loan Payback Prediction Web Application

## CIS 6005 - Computational Intelligence Project

### Overview
This Flask web application provides an intuitive interface for predicting loan payback probability using a trained Multi-Layer Perceptron (MLP) Neural Network model.

---

## 📁 Project Structure

```
loan_app/
├── app.py                      # Flask backend application
├── templates/
│   ├── index.html             # Home page with input form
│   ├── result.html            # Prediction results page
│   └── error.html             # Error handling page
├── static/                     # (Optional) CSS/JS files
├── loan_model.pkl             # Trained MLP model
├── scaler.pkl                 # Fitted StandardScaler
└── README.md                  # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Install Dependencies
```bash
pip install flask numpy scikit-learn joblib
```

### Step 2: Place Model Files
Ensure these files are in the `loan_app/` directory:
- `loan_model.pkl` (trained MLP model)
- `scaler.pkl` (fitted StandardScaler)

### Step 3: Run the Application
```bash
cd loan_app
python app.py
```

The application will start on: **http://localhost:5000**

---

## 💻 Usage

### Web Interface
1. Open browser and navigate to `http://localhost:5000`
2. Fill in the loan application form:
   - Financial details (income, loan amount, credit score, etc.)
   - Personal information (gender, marital status, education, etc.)
   - Loan details (purpose, grade)
3. Click "Predict Loan Payback"
4. View prediction result with confidence score and risk level

### API Endpoint

**Endpoint:** `POST /api/predict`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
    "annual_income": 45000,
    "debt_to_income_ratio": 0.35,
    "credit_score": 720,
    "loan_amount": 15000,
    "interest_rate": 12.5,
    "gender": 0,
    "marital_status": 1,
    "education_level": 1,
    "employment_status": 0,
    "loan_purpose": 0,
    "grade_subgrade": 4
}
```

**Response:**
```json
{
    "success": true,
    "prediction": 1,
    "prediction_text": "Loan Approved",
    "probability_default": 0.12,
    "probability_payback": 0.88,
    "confidence": 0.88
}
```

---

## 🎯 Model Information

- **Algorithm:** Multi-Layer Perceptron (MLP) Neural Network
- **Architecture:** 2 hidden layers (64, 32 neurons)
- **Activation Function:** ReLU
- **Training Accuracy:** 90.63%
- **Training Data:** 111,647 loan applications
- **Features:** 11 (5 numerical + 6 categorical)

---

## 📊 Feature Encoding

### Categorical Features (Label Encoded):

**Gender:**
- 0 = Female
- 1 = Male
- 2 = Other

**Marital Status:**
- 0 = Single
- 1 = Married
- 2 = Divorced
- 3 = Widowed

**Education Level:**
- 0 = High School
- 1 = Bachelor's
- 2 = Master's
- 3 = PhD
- 4 = Other

**Employment Status:**
- 0 = Employed
- 1 = Unemployed
- 2 = Self-employed
- 3 = Retired
- 4 = Student

**Loan Purpose:**
- 0 = Debt Consolidation
- 1 = Car Purchase
- 2 = Home Improvement
- 3 = Education
- 4 = Business
- 5 = Medical
- 6 = Vacation
- 7 = Other

---

## 🛠️ Technical Implementation

### Backend (Flask)
- **Route `/`:** Renders home page
- **Route `/predict`:** Handles form submission and prediction
- **Route `/api/predict`:** JSON API for programmatic access

### Preprocessing Pipeline
1. Extract form data
2. Create feature array (11 features)
3. Scale features using loaded StandardScaler
4. Pass to MLP model for prediction
5. Return probability and risk assessment

### Error Handling
- Input validation
- Model loading verification
- Exception catching with user-friendly messages

---

## 📸 Screenshots

### Home Page
Clean, intuitive input form for loan applications

### Result Page
- Color-coded prediction (Green = Approved, Red = High Risk)
- Confidence percentage
- Probability breakdown visualization
- Risk level indicator (LOW/MEDIUM/HIGH)
- Summary of input values

---

## 🔧 Troubleshooting

**Issue:** "Model not found"
- **Solution:** Ensure `loan_model.pkl` and `scaler.pkl` are in the same directory as `app.py`

**Issue:** "Feature mismatch"
- **Solution:** Verify categorical values are correctly encoded (0-based integers)

**Issue:** Port already in use
- **Solution:** Change port in `app.py`: `app.run(port=5001)`

---

## 📝 Development Notes

- Model was trained using Scikit-learn's MLPClassifier
- Preprocessing includes IQR outlier removal and standard scaling
- Class imbalance (80/20) handled during training
- Stratified train-test split maintained class distribution

---

## 👨‍💻 Author

**Pasindu Dilshan**  
Student ID: ST20274941  
Module: CIS 6005 - Computational Intelligence  
Institution: Cardiff Metropolitan University
=======
---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR-USERNAME/CIS6005-Loan-Prediction.git
cd CIS6005-Loan-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook
```bash
jupyter notebook notebook/loan_prediction.ipynb
```

### 4. Run Flask Application
```bash
cd flask_app
python app.py
```
Open browser: `http://localhost:5000`

---

## 📊 Dataset

**Source:** [Kaggle Playground Series S5E11](https://www.kaggle.com/competitions/playground-series-s5e11)

- **Records:** 593,994 loan applications
- **Features:** 12 (annual_income, credit_score, debt_to_income_ratio, etc.)
- **Target:** Binary (loan_paid_back: 0=Default, 1=Paid)
- **Class Distribution:** 80% paid-back, 20% default

---

## 🔍 Key Features

### Preprocessing
- **Outlier Removal:** IQR-based clipping for annual_income
- **Encoding:** LabelEncoder for categorical features
- **Scaling:** StandardScaler for numerical features
- **Train-Test Split:** 80-20 stratified split (475,195 train / 118,799 test)

### Model Implementation
- **Traditional ML:** Logistic Regression, Naive Bayes
- **Ensemble Methods:** Random Forest, XGBoost
- **Deep Learning:** MLP (2 hidden layers: 64, 32 neurons)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC for class discrimination
- Confusion Matrix analysis

---

## 🌐 Flask Web Application

Real-time loan prediction interface:
- Input loan details (income, credit score, etc.)
- Instant prediction (Approved/High Risk)
- Probability breakdown
- Risk level indicator (LOW/MEDIUM/HIGH)

**Screenshot:**
![Flask Application](images/flask_screenshot.png)

---

## 🏆 Kaggle Competition Results

**Public Score:** 0.9197 ROC-AUC  
**Private Score:** 0.9205 ROC-AUC

---

## 📝 Key Findings

1. **XGBoost Optimal:** Best performance (91.31%) with reasonable training time
2. **Ensemble Superiority:** Random Forest & XGBoost outperformed individual classifiers
3. **Deep Learning Competitive:** MLP matched ensemble methods but required 3x training time
4. **Feature Importance:** credit_score (28.3%), interest_rate (19.1%), debt_to_income_ratio (14.7%)
5. **Class Imbalance Handled:** Stratified sampling prevented majority-class bias

---

## 🔮 Future Improvements

- [ ] Implement SMOTE for better minority class handling
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Add SHAP for model explainability
- [ ] Deploy to cloud (AWS/Heroku)
- [ ] Create API endpoints for integration

---

## 👤 Author

**Pasindu Dilshan**  
Student ID: ST20274941  
Cardiff Metropolitan University  
BSc (Hons) Business Information Systems


---

## 📄 License

This project is submitted as academic coursework for CIS 6005.

---

## 🙏 Acknowledgements

- Cardiff Metropolitan University
- CIS 6005 Module Lecturer: Mr. Roy Ian
- Kaggle for providing the loan dataset
=======
# 🏦 Loan Payback Prediction Web Application

## CIS 6005 - Computational Intelligence Project

### Overview
This Flask web application provides an intuitive interface for predicting loan payback probability using a trained Multi-Layer Perceptron (MLP) Neural Network model.

---

## 📁 Project Structure

```
loan_app/
├── app.py                      # Flask backend application
├── templates/
│   ├── index.html             # Home page with input form
│   ├── result.html            # Prediction results page
│   └── error.html             # Error handling page
├── static/                     # (Optional) CSS/JS files
├── loan_model.pkl             # Trained MLP model
├── scaler.pkl                 # Fitted StandardScaler
└── README.md                  # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Install Dependencies
```bash
pip install flask numpy scikit-learn joblib
```

### Step 2: Place Model Files
Ensure these files are in the `loan_app/` directory:
- `loan_model.pkl` (trained MLP model)
- `scaler.pkl` (fitted StandardScaler)

### Step 3: Run the Application
```bash
cd loan_app
python app.py
```

The application will start on: **http://localhost:5000**

---

## 💻 Usage

### Web Interface
1. Open browser and navigate to `http://localhost:5000`
2. Fill in the loan application form:
   - Financial details (income, loan amount, credit score, etc.)
   - Personal information (gender, marital status, education, etc.)
   - Loan details (purpose, grade)
3. Click "Predict Loan Payback"
4. View prediction result with confidence score and risk level

### API Endpoint

**Endpoint:** `POST /api/predict`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
    "annual_income": 45000,
    "debt_to_income_ratio": 0.35,
    "credit_score": 720,
    "loan_amount": 15000,
    "interest_rate": 12.5,
    "gender": 0,
    "marital_status": 1,
    "education_level": 1,
    "employment_status": 0,
    "loan_purpose": 0,
    "grade_subgrade": 4
}
```

**Response:**
```json
{
    "success": true,
    "prediction": 1,
    "prediction_text": "Loan Approved",
    "probability_default": 0.12,
    "probability_payback": 0.88,
    "confidence": 0.88
}
```

---

## 🎯 Model Information

- **Algorithm:** Multi-Layer Perceptron (MLP) Neural Network
- **Architecture:** 2 hidden layers (64, 32 neurons)
- **Activation Function:** ReLU
- **Training Accuracy:** 90.63%
- **Training Data:** 111,647 loan applications
- **Features:** 11 (5 numerical + 6 categorical)

---

## 📊 Feature Encoding

### Categorical Features (Label Encoded):

**Gender:**
- 0 = Female
- 1 = Male
- 2 = Other

**Marital Status:**
- 0 = Single
- 1 = Married
- 2 = Divorced
- 3 = Widowed

**Education Level:**
- 0 = High School
- 1 = Bachelor's
- 2 = Master's
- 3 = PhD
- 4 = Other

**Employment Status:**
- 0 = Employed
- 1 = Unemployed
- 2 = Self-employed
- 3 = Retired
- 4 = Student

**Loan Purpose:**
- 0 = Debt Consolidation
- 1 = Car Purchase
- 2 = Home Improvement
- 3 = Education
- 4 = Business
- 5 = Medical
- 6 = Vacation
- 7 = Other

---

## 🛠️ Technical Implementation

### Backend (Flask)
- **Route `/`:** Renders home page
- **Route `/predict`:** Handles form submission and prediction
- **Route `/api/predict`:** JSON API for programmatic access

### Preprocessing Pipeline
1. Extract form data
2. Create feature array (11 features)
3. Scale features using loaded StandardScaler
4. Pass to MLP model for prediction
5. Return probability and risk assessment

### Error Handling
- Input validation
- Model loading verification
- Exception catching with user-friendly messages

---

## 📸 Screenshots

### Home Page
Clean, intuitive input form for loan applications

### Result Page
- Color-coded prediction (Green = Approved, Red = High Risk)
- Confidence percentage
- Probability breakdown visualization
- Risk level indicator (LOW/MEDIUM/HIGH)
- Summary of input values

---

## 🔧 Troubleshooting

**Issue:** "Model not found"
- **Solution:** Ensure `loan_model.pkl` and `scaler.pkl` are in the same directory as `app.py`

**Issue:** "Feature mismatch"
- **Solution:** Verify categorical values are correctly encoded (0-based integers)

**Issue:** Port already in use
- **Solution:** Change port in `app.py`: `app.run(port=5001)`

---

## 📝 Development Notes

- Model was trained using Scikit-learn's MLPClassifier
- Preprocessing includes IQR outlier removal and standard scaling
- Class imbalance (80/20) handled during training
- Stratified train-test split maintained class distribution

---

## 👨‍💻 Author

**Pasindu Dilshan**  
Student ID: ST20274941  
Module: CIS 6005 - Computational Intelligence  
Institution: Cardiff Metropolitan University

---

## 📄 License

This project is submitted as academic coursework for CIS 6005.

---

## 🙏 Acknowledgements

- Cardiff Metropolitan University
- CIS 6005 Module Lecturer: Mr. Roy Ian
- Kaggle for providing the loan dataset

