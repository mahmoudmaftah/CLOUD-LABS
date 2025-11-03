import os
import joblib
import numpy as np
from flask import Flask, render_template, request

from train_model import train_and_save

ARTIFACTS = os.path.join(os.path.dirname(__file__), 'model_artifacts.joblib')

app = Flask(__name__)


def load_artifacts():
    if not os.path.exists(ARTIFACTS):
        print('Artifacts not found. Training model...')
        train_and_save(artifacts_path=ARTIFACTS)
    return joblib.load(ARTIFACTS)


@app.route('/', methods=['GET', 'POST'])
def index():
    artifacts = load_artifacts()
    features = artifacts['features']
    means = artifacts['feature_means']

    if request.method == 'POST':
        # Collect feature values from form
        vals = []
        for f in features:
            v = request.form.get(f)
            try:
                vals.append(float(v))
            except Exception:
                vals.append(float(means.get(f, 0.0)))

        X = np.array(vals).reshape(1, -1)
        X_scaled = artifacts['scaler'].transform(X)
        model = artifacts['model']
        pred_idx = int(model.predict(X_scaled)[0])
        pred_name = artifacts['target_names'][pred_idx]
        probs = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)[0].tolist()

        return render_template('result.html', prediction=pred_name, probs=probs, target_names=artifacts['target_names'])

    # GET
    return render_template('index.html', features=features, means=means)


if __name__ == '__main__':
    # Use debug for development
    app.run(host='0.0.0.0', port=5000, debug=True)
