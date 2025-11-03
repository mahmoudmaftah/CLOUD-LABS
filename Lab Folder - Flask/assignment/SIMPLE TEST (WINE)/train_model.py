import json
from pprint import pprint
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_and_save(artifacts_path='model_artifacts.joblib', top_n=5, random_state=42):
    """Load dataset, preprocess, select top features, compare models and save best model + metadata."""
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    # 1) Basic missing value treatment (dataset has none but we make it robust)
    X = X.fillna(X.median())

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 3) Use a RandomForest to compute feature importances and select top features
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = list(importances.sort_values(ascending=False).head(top_n).index)

    # Keep only selected features
    X_train_sel = X_train[top_features]
    X_test_sel = X_test[top_features]

    # 4) Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # 5) Compare models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=random_state),
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(random_state=random_state)
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        acc = m.score(X_test_scaled, y_test)
        results[name] = {'model': m, 'accuracy': acc}

    # Pick best model by accuracy
    best_name = max(results.keys(), key=lambda n: results[n]['accuracy'])
    best_model = results[best_name]['model']
    best_acc = results[best_name]['accuracy']

    # Save artifacts
    artifacts = {
        'model': best_model,
        'scaler': scaler,
        'features': top_features,
        'feature_means': X[top_features].mean().to_dict(),
        'target_names': list(data.target_names),
        'metrics': {n: results[n]['accuracy'] for n in results}
    }

    joblib.dump(artifacts, artifacts_path)

    print('Training complete. Best model:', best_name)
    print('Test accuracy: {:.4f}'.format(best_acc))
    print('Selected features:')
    pprint(top_features)
    print('\nModel accuracies:')
    pprint(artifacts['metrics'])
    return artifacts


if __name__ == '__main__':
    train_and_save()
