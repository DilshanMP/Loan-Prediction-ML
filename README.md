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

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- Cardiff Metropolitan University - ICBT Campus
- Mr. Roy Ian - Module Leader
- Kaggle Community - Dataset provision
- Open-source contributors - Python libraries

---

## 📧 Contact

- **Email:** your.email@example.com
- **LinkedIn:** [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **GitHub:** [Your GitHub](https://github.com/yourusername)

---

**⭐ If you find this project helpful, please give it a star!**
