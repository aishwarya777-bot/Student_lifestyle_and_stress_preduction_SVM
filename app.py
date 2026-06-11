import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# --- Model Loading Configuration ---
# Configured directly for your SVM model file
MODEL_PATH = "SVM_model.pkl" 

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Model load alert: {e}")

# The 8 exact features your SVM model expects in correct order
FEATURE_KEYS = [
    'Student_Type', 'Sleep_Hours', 'Study_Hours', 'Social_Media_Hours', 
    'Attendance', 'Exam_Pressure', 'Family_Support', 'Month'
]

# --- Premium Glassmorphism UI Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Analytics Workspace</title>
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
        <h1>SVM Analysis Workspace</h1>
        <p>Interactive diagnostics analytics framework engine</p>
    </header>

    <form method="POST" action="/">
        <div class="form-grid">
            
            <div class="field-card">
                <div class="label-wrapper"><label>Profile Category</label></div>
                <select name="Student_Type">
                    <option value="0" {% if form_values.get('Student_Type') == '0' %}selected{% endif %}>Standard Baseline Profile</option>
                    <option value="1" {% if form_values.get('Student_Type') == '1' %}selected{% endif %}>Alternative Profile</option>
                </select>
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Sleep Duration (Hours)</label>
                    <span class="live-value" id="val_Sleep_Hours">7</span>
                </div>
                <input type="range" name="Sleep_Hours" min="1" max="24" value="{{ form_values.get('Sleep_Hours', 7) }}" oninput="document.getElementById('val_Sleep_Hours').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Study Allocation (Hours)</label>
                    <span class="live-value" id="val_Study_Hours">4</span>
                </div>
                <input type="range" name="Study_Hours" min="0" max="24" value="{{ form_values.get('Study_Hours', 4) }}" oninput="document.getElementById('val_Study_Hours').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Screen / Social Engagement (Hours)</label>
                    <span class="live-value" id="val_Social_Media_Hours">2</span>
                </div>
                <input type="range" name="Social_Media_Hours" min="0" max="24" value="{{ form_values.get('Social_Media_Hours', 2) }}" oninput="document.getElementById('val_Social_Media_Hours').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Attendance Level (%)</label>
                    <span class="live-value" id="val_Attendance">85</span>
                </div>
                <input type="range" name="Attendance" min="0" max="100" value="{{ form_values.get('Attendance', 85) }}" oninput="document.getElementById('val_Attendance').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Stress & Pressure Index</label>
                    <span class="live-value" id="val_Exam_Pressure">5</span>
                </div>
                <input type="range" name="Exam_Pressure" min="1" max="10" value="{{ form_values.get('Exam_Pressure', 5) }}" oninput="document.getElementById('val_Exam_Pressure').innerText=this.value">
            </div>

            <div class="field-card">
                <div class="label-wrapper"><label>Support Environment Baseline</label></div>
                <select name="Family_Support">
                    <option value="0" {% if form_values.get('Family_Support') == '0' %}selected{% endif %}>Standard Baseline Environment</option>
                    <option value="1" {% if form_values.get('Family_Support') == '1' %}selected{% endif %}>High Support Environment</option>
                </select>
            </div>

            <div class="field-card">
                <div class="label-wrapper">
                    <label>Timeline Interval (Month)</label>
                    <span class="live-value" id="val_Month">6</span>
                </div>
                <input type="range" name="Month" min="1" max="12" value="{{ form_values.get('Month', 6) }}" oninput="document.getElementById('val_Month').innerText=this.value">
            </div>

            <div class="action-row">
                <button type="submit" class="submit-btn">Execute Diagnostic Predictor Pipeline</button>
            </div>
        </div>
    </form>

    {% if prediction_label %}
        <div class="result-panel {% if 'HIGH RISK' in prediction_label or 'ALERT' in prediction_label %}status-positive{% else %}status-negative{% endif %}">
            <div class="result-title">Inference Engine Diagnostic Conclusion</div>
            <div class="result-value">{{ prediction_label }}</div>
        </div>
    {% endif %}
</div>

<script>
    window.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('input[type="range"]').forEach(input => {
            const element = document.getElementById('val_' + input.name);
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
            prediction_label = f"Model Error: System core execution binary file '{MODEL_PATH}' missing or unreadable."
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
                prediction_label = "⚠️ ALERT CLASSIFICATION DETECTED (Value: 1)"
            else:
                prediction_label = "✅ NORMAL BASELINE STANDARD CONFIRMED (Value: 0)"
                
        except Exception as e:
            prediction_label = f"Analysis Interrupted: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_label=prediction_label, form_values=form_values)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
