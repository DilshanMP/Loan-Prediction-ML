from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import traceback

app = Flask(__name__)

# Load the trained model and scaler
try:
    scaler = joblib.load('../models/scaler.pkl')
    model = joblib.load('../models/loan_model.pkl')
    print("✓ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model or scaler: {e}")
    raise

@app.route('/')
def home():
    """Render the home page with the input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        # Numerical features
        annual_income = float(request.form['annual_income'])
        debt_to_income_ratio = float(request.form['debt_to_income_ratio'])
        credit_score = float(request.form['credit_score'])
        loan_amount = float(request.form['loan_amount'])
        interest_rate = float(request.form['interest_rate'])
        term = int(request.form['term'])  # ADDED: Missing feature!
        
        # Categorical features (Label Encoded)
        gender = int(request.form['gender'])
        marital_status = int(request.form['marital_status'])
        education_level = int(request.form['education_level'])
        employment_status = int(request.form['employment_status'])
        loan_purpose = int(request.form['loan_purpose'])
        grade_subgrade = int(request.form['grade_subgrade'])
        
        # Create feature array in EXACT order as training data
        # Order should match: annual_income, debt_to_income_ratio, credit_score, 
        # loan_amount, interest_rate, term, gender, marital_status, 
        # education_level, employment_status, loan_purpose, grade_subgrade
        features = np.array([[
            annual_income,
            debt_to_income_ratio,
            credit_score,
            loan_amount,
            interest_rate,
            term,
            gender,
            marital_status,
            education_level,
            employment_status,
            loan_purpose,
            grade_subgrade
        ]])
        
        print(f"Features shape: {features.shape}")  # Debug: Should be (1, 12)
        print(f"Features: {features}")
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'prediction_text': 'LOAN APPROVED - Likely to Pay Back' if prediction == 1 else 'HIGH RISK - Likely to Default',
            'probability_default': float(probability[0]) * 100,
            'probability_payback': float(probability[1]) * 100,
            'confidence': float(max(probability)) * 100,
            'risk_level': 'LOW' if probability[1] > 0.7 else 'MEDIUM' if probability[1] > 0.5 else 'HIGH',
            
            # Include input data for display
            'input_data': {
                'annual_income': f"Rs {annual_income:,.2f}",
                'loan_amount': f"Rs {loan_amount:,.2f}",
                'credit_score': credit_score,
                'debt_to_income_ratio': f"{debt_to_income_ratio:.2f}",
                'interest_rate': f"{interest_rate:.2f}%",
                'term': f"{term} months",
                'gender': 'Male' if gender == 1 else 'Female',
                'marital_status': ['Single', 'Married', 'Divorced'][marital_status] if marital_status < 3 else 'Other',
                'education_level': ['High School', "Bachelor's", "Master's", 'PhD'][education_level] if education_level < 4 else 'Other',
                'employment_status': ['Employed', 'Self-Employed', 'Unemployed'][employment_status] if employment_status < 3 else 'Other',
                'loan_purpose': ['Debt Consolidation', 'Home Improvement', 'Business', 'Education', 'Other'][loan_purpose] if loan_purpose < 5 else 'Other',
                'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G'][grade_subgrade // 5] if grade_subgrade < 35 else 'G'
            }
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        print(f"Error during prediction: {error_message}")
        print(error_traceback)
        return render_template('error.html', error=error_message)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON response)"""
    try:
        data = request.get_json()
        
        # Extract features in correct order
        features = np.array([[
            float(data['annual_income']),
            float(data['debt_to_income_ratio']),
            float(data['credit_score']),
            float(data['loan_amount']),
            float(data['interest_rate']),
            int(data['term']),
            int(data['gender']),
            int(data['marital_status']),
            int(data['education_level']),
            int(data['employment_status']),
            int(data['loan_purpose']),
            int(data['grade_subgrade'])
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_text': 'Approved' if prediction == 1 else 'Rejected',
            'probability_default': float(probability[0]),
            'probability_payback': float(probability[1]),
            'confidence': float(max(probability))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏦 LOAN PAYBACK PREDICTION SYSTEM")
    print("="*60)
    print("✓ Server starting on http://localhost:5000")
    print("✓ Press CTRL+C to quit")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
