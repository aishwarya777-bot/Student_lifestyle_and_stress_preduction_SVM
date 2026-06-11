import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load your SVM model
MODEL_PATH = "SVM_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model file: {e}")
    model = None

# Automatically detect how many features your SVM model expects
if model and hasattr(model, "n_features_in_"):
    NUM_FEATURES = model.n_features_in_
elif model and hasattr(model, "support_vectors_") and len(model.support_vectors_) > 0:
    NUM_FEATURES = model.support_vectors_.shape[1]
else:
    # Safe default fallback if structural inspection isn't exposed
    NUM_FEATURES = 4  

# Attractive, responsive HTML UI Layout embedded directly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #f8fafc;
            --panel: #ffffff;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text-dark);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            max-width: 750px;
            width: 100%;
        }
        .card {
            background: var(--panel);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border);
        }
        h1 {
            font-size: 26px;
            font-weight: 700;
            margin: 0 0 8px 0;
            text-align: center;
            color: var(--text-dark);
        }
        .subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 35px;
            font-size: 14px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        @media (max-width: 550px) {
            .form-grid { grid-template-columns: 1fr; }
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input {
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 15px;
            background-color: #fbfbfb;
            transition: all 0.2s ease;
        }
        input:focus {
            outline: none;
            border-color: var(--primary);
            background-color: #fff;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        .btn-container {
            grid-column: span 2;
            margin-top: 15px;
        }
        @media (max-width: 550px) { .btn-container { grid-column: span 1; } }
        .btn-submit {
            width: 100%;
            background: var(--primary);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-submit:hover { background: var(--primary-hover); }
        
        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            animation: fadeIn 0.4s ease;
        }
        .cat-positive {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
        .cat-negative {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="card">
        <h1>SVM Analysis Dashboard</h1>
        <p class="subtitle">Provide the numeric metric attributes required by the Support Vector Machine framework.</p>
        
        <form method="POST" action="/">
            <div class="form-grid">
                {% for i in range(num_features) %}
                <div class="form-group">
                    <label>Feature Input Metric {{ i + 1 }}</label>
                    <input type="number" name="feature_{{ i }}" step="any" required placeholder="0.00">
                </div>
                {% endfor %}
                
                <div class="btn-container">
                    <button type="submit" class="btn-submit">Process Predictive Inference</button>
                </div>
            </div>
        </form>

        {% if categorical_prediction %}
            <div class="result-box {% if 'Alert' in categorical_prediction or 'Positive' in categorical_prediction %}cat-positive{% else %}cat-negative{% endif %}">
                System Output: {{ categorical_prediction }}
            </div>
        {% endif %}
    </div>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    categorical_prediction = None
    
    if request.method == "POST":
        if model is None:
            return render_template_string(HTML_TEMPLATE, num_features=NUM_FEATURES, categorical_prediction="Model file failed to load correctly.")
        
        try:
            # Build input vector in correct index format dynamically
            input_vector = []
            for i in range(NUM_FEATURES):
                val = float(request.form[f"feature_{i}"])
                input_vector.append(val)
                
            # Shape into a 2D array matrix for scikit-learn
            final_features = np.array([input_vector])
            
            # Run model execution
            raw_prediction = model.predict(final_features)[0]
            
            # Map numeric outputs into categorical classifications
            if int(raw_prediction) == 1:
                categorical_prediction = "High Category Verified (Value: 1 / Alert Status)"
            else:
                categorical_prediction = "Normal Category Verified (Value: 0 / Standard Baseline)"
                
        except Exception as e:
            categorical_prediction = f"Error during optimization/parsing: {str(e)}"

    return render_template_string(HTML_TEMPLATE, num_features=NUM_FEATURES, categorical_prediction=categorical_prediction)

if __name__ == "__main__":
    # Dynamically bind port handling for Render environment routing
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
