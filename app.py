import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# --- Model Loading Configuration ---
# Update this filename based on which model you are executing ('logistic_pkl.pkl' or 'SVM_model.pkl')
MODEL_PATH = "logistic_pkl.pkl" 

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Model load alert: {e}")

# Exact explicit feature alignment match required by the dataset schema
FEATURE_KEYS = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
    'ejection_fraction', 'high_blood_pressure', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

# --- Premium Modern UI Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Analytics Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --accent-primary: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.15);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --success-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.2) 100%);
            --success-border: rgba(16, 185, 129, 0.3);
            --danger-bg: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.2) 100%);
            --danger-border: rgba(239, 68, 68, 0.3);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px 20px;
        }

        .dashboard {
            max-width: 950px;
            width: 100%;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 45px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        header h1 {
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 10px;
            background: linear-gradient(to right, #fff, #c7d2fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        header p {
            color: var(--text-secondary);
            font-size: 15px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 28px;
        }

        @media (max-width: 768px) {
            .form-grid { grid-template-columns: 1fr; }
        }

        .field-card {
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--glass-border);
            padding: 20px;
            border-radius: 14px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .field-card:focus-within {
            border-color: var(--accent-primary);
            box-shadow: 0 0 20px var(--accent-glow);
            background: rgba(255, 255, 255, 0.02);
        }

        .label-wrapper {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .live-value {
            font-size: 14px;
            font-weight: 700;
            color: var(--accent-primary);
            background: rgba(99, 102, 241, 0.1);
            padding: 2px 8px;
            border-radius: 6px;
        }

        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: var(--input-bg);
            border-radius: 4px;
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--accent-primary);
            cursor: pointer;
            transition: transform 0.1s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }

        select {
            width: 100%;
            padding: 12px;
            background: var(--input-bg);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 15px;
            outline: none;
            cursor: pointer;
        }

        select option {
            background: #1e1b4b;
            color: var(--text-primary);
        }

        .action-row {
            grid-column: span 2;
            margin-top: 20px;
        }

        @media (max-width: 768px) { .action-row { grid-column: span 1; } }

        .submit-btn {
            width: 100%;
            background: var(--accent-primary);
            color: #fff;
            padding: 16px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 15px var(--accent-glow);
            transition: all 0.2s ease;
        }

        .submit-btn:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .result-panel {
            margin-top: 40px;
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .status-positive {
            background: var(--danger-bg);
            border: 1px solid var(--danger-border);
            color: #fca5a5;
        }

        .status-negative {
            background: var(--success-bg);
            border: 1px solid var(--success-border);
            color: #6ee7b7;
        }

        .result-title {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
            opacity: 0.8;
        }

        .result-value {
            font-size: 22px;
            font-weight: 700;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="dashboard">
    <header>
        <h1>Clinical Predictive Workspace</h1>
        <p>Interactive diagnostics analytics framework engine</p>
    </header>

    <form method="POST" action="/">
        <div class="form-grid">
            
            <div class="field-card">
                <div class="label-wrapper">
                    <label>Age Profile</label>
                    <span class="live-value" id="val_age">55</span>
                </div>
                <input type="range" name="age" min="1" max="110" value="{{ form_values.get('age', 55) }}" oninput="document.getElementById('val_age').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper"><label>Anaemia Condition</label></div>
                <select name="anaemia">
                    <option value="0" {% if form_values.get('anaemia') == '0' %}selected{% endif %}>Negative / Normal Baseline</option>
                    <option value="1" {% if form_values.get('anaemia') == '1' %}selected{% endif %}>Positive Case Detected</option>
                </select>
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Creatinine Phosphokinase (mcg/L)</label>
                    <span class="live-value" id="val_cpk">250</span>
                </div>
                <input type="range" name="creatinine_phosphokinase" min="10" max="8000" value="{{ form_values.get('creatinine_phosphokinase', 250) }}" oninput="document.getElementById('val_cpk').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper"><label>Diabetes History</label></div>
                <select name="diabetes">
                    <option value="0" {% if form_values.get('diabetes') == '0' %}selected{% endif %}>Non-Diabetic</option>
                    <option value="1" {% if form_values.get('diabetes') == '1' %}selected{% endif %}>Diabetic Clinical Diagnosis</option>
                </select>
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Ejection Fraction (%)</label>
                    <span class="live-value" id="val_ef">40</span>
                </div>
                <input type="range" name="ejection_fraction" min="10" max="80" value="{{ form_values.get('ejection_fraction', 40) }}" oninput="document.getElementById('val_ef').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper"><label>Hypertension Status</label></div>
                <select name="high_blood_pressure">
                    <option value="0" {% if form_values.get('high_blood_pressure') == '0' %}selected{% endif %}>Normal / Controlled Blood Pressure</option>
                    <option value="1" {% if form_values.get('high_blood_pressure') == '1' %}selected{% endif %}>Hypertensive State Confirmed</option>
                </select>
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Platelets Count (kiloplatelets/mL)</label>
                    <span class="live-value" id="val_plat">250000</span>
                </div>
                <input type="range" name="platelets" min="25000" max="850000" step="5000" value="{{ form_values.get('platelets', 250000) }}" oninput="document.getElementById('val_plat').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Serum Creatinine (mg/dL)</label>
                    <span class="live-value" id="val_creat">1.2</span>
                </div>
                <input type="range" name="serum_creatinine" min="0.1" max="10.0" step="0.1" value="{{ form_values.get('serum_creatinine', 1.2) }}" oninput="document.getElementById('val_creat').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Serum Sodium (mEq/L)</label>
                    <span class="live-value" id="val_sod">135</span>
                </div>
                <input type="range" name="serum_sodium" min="100" max="150" value="{{ form_values.get('serum_sodium', 135) }}" oninput="document.getElementById('val_sod').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper"><label>Biological Sex Classification</label></div>
                <select name="sex">
                    <option value="0" {% if form_values.get('sex') == '0' %}selected{% endif %}>Female</option>
                    <option value="1" {% if form_values.get('sex') == '1' %}selected{% endif %}>Male</option>
                </select>
            </div>

            <div class="field-card">
                <div class="label-wrapper"><label>Tobacco Dependency Profile</label></div>
                <select name="smoking">
                    <option value="0" {% if form_values.get('smoking') == '0' %}selected{% endif %}>Identified Non-Smoker</option>
                    <option value="1" {% if form_values.get('smoking') == '1' %}selected{% endif %}>Identified Active Smoker</option>
                </select>
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Clinical Observation Period (Days)</label>
                    <span class="live-value" id="val_time">100</span>
                </div>
                <input type="range" name="time" min="1" max="300" value="{{ form_values.get('time', 100) }}" oninput="document.getElementById('val_time').innerText=this.value">
            </div>

            <div class="action-row">
                <button type="submit" class="submit-btn">Execute Diagnostic Predictor Pipeline</button>
            </div>
        </div>
    </form>

    {% if prediction_label %}
        <div class="result-panel {% if 'HIGH RISK' in prediction_label %}status-positive{% else %}status-negative{% endif %}">
            <div class="result-title">Inference Engine Diagnostic Conclusion</div>
            <div class="result-value">{{ prediction_label }}</div>
        </div>
    {% endif %}
</div>

<script>
    // System script synchronization logic ensuring initial sliders render their values dynamically
    window.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('input[type="range"]').forEach(input => {
            const displayId = 'val_' + input.name.replace('creatinine_phosphokinase', 'cpk').replace('ejection_fraction', 'ef').replace('platelets', 'plat').replace('serum_creatinine', 'creat').replace('serum_sodium', 'sod');
            const element = document.getElementById(displayId);
            if(element) element.innerText = input.value;
        });
    });
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_label = None
    form_values = {}
    
    if request.method == "POST":
        form_values = request.form.to_dict()
        if model is None:
            prediction_label = "Model Error: System core execution binary file missing or unreadable."
            return render_template_string(HTML_TEMPLATE, prediction_label=prediction_label, form_values=form_values)
        
        try:
            # Construct feature payload array following model criteria alignment
            payload = []
            for key in FEATURE_KEYS:
                payload.append(float(request.form[key]))
                
            final_features = np.array([payload])
            raw_output = model.predict(final_features)[0]
            
            # Map specific numeric outputs directly into readable categorical alerts
            if int(raw_output) == 1:
                prediction_label = "⚠️ HIGH RISK ALERT STATUS CLASSIFICATION"
            else:
                prediction_label = "✅ LOW RISK BASELINE STANDARD STATUS"
                
        except Exception as e:
            prediction_label = f"Analysis Interrupted: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_label=prediction_label, form_values=form_values)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
