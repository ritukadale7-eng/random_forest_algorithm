import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model from random.pkl
with open('random.pkl', 'rb') as f:
    model = pickle.load(f)

# Define HTML template with modern styling and smooth animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Layoff & Workforce Risk Analyzer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-color: #f8fafc;
            --border-color: rgba(255, 255, 255, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 850px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }

        p.subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: #cbd5e1;
        }

        input, select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3);
        }

        button {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.85rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, opacity 0.2s ease;
        }

        button:hover {
            opacity: 0.95;
            transform: translateY(-2px);
        }

        .result-card {
            margin-top: 2rem;
            background: rgba(15, 23, 42, 0.8);
            border-left: 5px solid var(--primary);
            padding: 1.5rem;
            border-radius: 12px;
            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-card h3 {
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            color: #818cf8;
        }

        .result-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #fff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Workforce Risk & AI Impact Predictor</h1>
        <p class="subtitle">Powered by Random Forest Machine Learning Model</p>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label>Company Size (Employees)</label>
                    <input type="number" name="company_size" required placeholder="e.g., 5000">
                </div>
                <div class="form-group">
                    <label>Month</label>
                    <input type="number" name="month" min="1" max="12" required placeholder="e.g., 6">
                </div>
                <div class="form-group">
                    <label>Year</label>
                    <input type="number" name="year" required placeholder="e.g., 2026">
                </div>
                <div class="form-group">
                    <label>Layoffs Count</label>
                    <input type="number" name="layoffs_count" required placeholder="e.g., 150">
                </div>
                <div class="form-group">
                    <label>Layoff Percentage (%)</label>
                    <input type="number" step="any" name="layoff_percentage" required placeholder="e.g., 5.5">
                </div>
                <div class="form-group">
                    <label>AI Automation Impact (0-10)</label>
                    <input type="number" step="any" name="ai_automation_impact" required placeholder="e.g., 7.2">
                </div>
                <div class="form-group">
                    <label>AI Replacement Risk (0-10)</label>
                    <input type="number" step="any" name="ai_replacement_risk" required placeholder="e.g., 6.0">
                </div>
                <div class="form-group">
                    <label>Open Roles</label>
                    <input type="number" name="open_roles" required placeholder="e.g., 25">
                </div>
                <div class="form-group">
                    <label>Remote Jobs Percentage (%)</label>
                    <input type="number" step="any" name="remote_jobs_percentage" required placeholder="e.g., 40">
                </div>
                <div class="form-group">
                    <label>Stock Growth Percent (%)</label>
                    <input type="number" step="any" name="stock_growth_percent" required placeholder="e.g., 12.5">
                </div>
                <div class="form-group">
                    <label>Revenue Growth Percent (%)</label>
                    <input type="number" step="any" name="revenue_growth_percent" required placeholder="e.g., 8.1">
                </div>
                <div class="form-group">
                    <label>Salary Budget Change (%)</label>
                    <input type="number" step="any" name="salary_budget_change" required placeholder="e.g., 2.0">
                </div>
                <div class="form-group">
                    <label>AI Adoption Level (0-10)</label>
                    <input type="number" step="any" name="ai_adoption_level" required placeholder="e.g., 8.0">
                </div>
                <div class="form-group">
                    <label>Employee Sentiment (0-10)</label>
                    <input type="number" step="any" name="employee_sentiment" required placeholder="e.g., 6.5">
                </div>
                <div class="form-group">
                    <label>Job Security Score (0-10)</label>
                    <input type="number" step="any" name="job_security_score" required placeholder="e.g., 7.0">
                </div>
            </div>
            <button type="submit">Run Prediction Analysis</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Analysis Result</h3>
            <div class="result-value">Predicted Outcome: {{ prediction }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract numerical features matching the model's expected input order
            features = [
                float(request.form['company_size']),
                float(request.form['month']),
                float(request.form['year']),
                float(request.form['layoffs_count']),
                float(request.form['layoff_percentage']),
                float(request.form['ai_automation_impact']),
                float(request.form['ai_replacement_risk']),
                float(request.form['open_roles']),
                float(request.form['remote_jobs_percentage']),
                float(request.form['stock_growth_percent']),
                float(request.form['revenue_growth_percent']),
                float(request.form['salary_budget_change']),
                float(request.form['ai_adoption_level']),
                float(request.form['employee_sentiment']),
                float(request.form['job_security_score'])
            ]
            
            # Reshape for model prediction
            input_data = np.array([features])
            pred = model.predict(input_data)
            prediction = str(pred[0])
        except Exception as e:
            prediction = f"Error during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)