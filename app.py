import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'random.pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Embedded HTML Template with vibrant styling & 21 input fields
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Condition & Layoff Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: #ffffff;
            width: 100%;
            max-width: 900px;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            max-height: 90vh;
            overflow-y: auto;
        }
        h1 {
            color: #333333;
            margin-bottom: 8px;
            font-size: 26px;
            text-align: center;
        }
        p.subtitle {
            text-align: center;
            color: #666666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
        }
        .input-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 12px;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input {
            padding: 10px 14px;
            border: 1px solid #cbd5e0;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
            outline: none;
        }
        .btn-container {
            grid-column: 1 / -1;
            margin-top: 10px;
        }
        button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: opacity 0.3s ease;
        }
        button:hover {
            opacity: 0.9;
        }
        .result-box {
            margin-top: 25px;
            padding: 15px;
            background: #edf2f7;
            border-left: 5px solid #667eea;
            border-radius: 6px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            color: #2d3748;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Market & Workforce Predictor</h1>
        <p class="subtitle">Powered by Random Forest Classifier</p>
        
        <form action="/predict" method="POST">
            <div class="form-grid">
                <div class="input-group">
                    <label>Company Size</label>
                    <input type="number" step="any" name="company_size" required value="5000">
                </div>
                <div class="input-group">
                    <label>Month</label>
                    <input type="number" name="month" min="1" max="12" required value="10">
                </div>
                <div class="input-group">
                    <label>Year</label>
                    <input type="number" name="year" required value="2026">
                </div>
                <div class="input-group">
                    <label>Layoffs Count</label>
                    <input type="number" step="any" name="layoffs_count" required value="150">
                </div>
                <div class="input-group">
                    <label>Layoff Percentage</label>
                    <input type="number" step="any" name="layoff_percentage" required value="0.05">
                </div>
                <div class="input-group">
                    <label>AI Automation Impact</label>
                    <input type="number" step="any" name="ai_automation_impact" required value="3">
                </div>
                <div class="input-group">
                    <label>AI Replacement Risk</label>
                    <input type="number" step="any" name="ai_replacement_risk" required value="2">
                </div>
                <div class="input-group">
                    <label>Open Roles</label>
                    <input type="number" step="any" name="open_roles" required value="45">
                </div>
                <div class="input-group">
                    <label>Remote Jobs %</label>
                    <input type="number" step="any" name="remote_jobs_percentage" required value="30">
                </div>
                <div class="input-group">
                    <label>Stock Growth %</label>
                    <input type="number" step="any" name="stock_growth_percent" required value="12.5">
                </div>
                <div class="input-group">
                    <label>Revenue Growth %</label>
                    <input type="number" step="any" name="revenue_growth_percent" required value="8.0">
                </div>
                <div class="input-group">
                    <label>Salary Budget Change</label>
                    <input type="number" step="any" name="salary_budget_change" required value="2.5">
                </div>
                <div class="input-group">
                    <label>AI Adoption Level</label>
                    <input type="number" step="any" name="ai_adoption_level" required value="4">
                </div>
                <div class="input-group">
                    <label>Employee Sentiment</label>
                    <input type="number" step="any" name="employee_sentiment" required value="3.5">
                </div>
                <div class="input-group">
                    <label>Job Security Score</label>
                    <input type="number" step="any" name="job_security_score" required value="7.0">
                </div>
                <div class="input-group">
                    <label>Inflation Rate %</label>
                    <input type="number" step="any" name="inflation_rate" required value="3.2">
                </div>
                <div class="input-group">
                    <label>Interest Rate %</label>
                    <input type="number" step="any" name="interest_rate" required value="5.0">
                </div>
                <div class="input-group">
                    <label>Venture Funding ($M)</label>
                    <input type="number" step="any" name="venture_funding_millions" required value="25.0">
                </div>
                <div class="input-group">
                    <label>Attrition Rate %</label>
                    <input type="number" step="any" name="attrition_rate" required value="10.5">
                </div>
                <div class="input-group">
                    <label>Hiring Freeze Flag (0 or 1)</label>
                    <input type="number" name="hiring_freeze_flag" min="0" max="1" required value="0">
                </div>
                <div class="input-group">
                    <label>Productivity Index</label>
                    <input type="number" step="any" name="productivity_index" required value="85.0">
                </div>
                
                <div class="btn-container">
                    <button type="submit">Run Prediction</button>
                </div>
            </div>
        </form>

        {% if prediction_text %}
            <div class="result-box">
                {{ prediction_text }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction_text="Error: Model not loaded properly.")
    
    try:
        # Collect values from form inputs
        form_vals = [
            float(request.form.get('company_size', 0)),
            float(request.form.get('month', 1)),
            float(request.form.get('year', 2026)),
            float(request.form.get('layoffs_count', 0)),
            float(request.form.get('layoff_percentage', 0.0)),
            float(request.form.get('ai_automation_impact', 0)),
            float(request.form.get('ai_replacement_risk', 0)),
            float(request.form.get('open_roles', 0)),
            float(request.form.get('remote_jobs_percentage', 0)),
            float(request.form.get('stock_growth_percent', 0)),
            float(request.form.get('revenue_growth_percent', 0)),
            float(request.form.get('salary_budget_change', 0)),
            float(request.form.get('ai_adoption_level', 0)),
            float(request.form.get('employee_sentiment', 0)),
            float(request.form.get('job_security_score', 0)),
            float(request.form.get('inflation_rate', 0)),
            float(request.form.get('interest_rate', 0)),
            float(request.form.get('venture_funding_millions', 0)),
            float(request.form.get('attrition_rate', 0)),
            float(request.form.get('hiring_freeze_flag', 0)),
            float(request.form.get('productivity_index', 0))
        ]
        
        features = np.array(form_vals)
        
        # Automatically match the exact number of features expected by the loaded model
        expected_n_features = getattr(model, "n_features_in_", len(features))
        if len(features) < expected_n_features:
            # Pad with zeros if fewer
            features = np.pad(features, (0, expected_n_features - len(features)), 'constant')
        elif len(features) > expected_n_features:
            # Trim if extra
            features = features[:expected_n_features]
            
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Predicted Market Condition: {prediction}")
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)