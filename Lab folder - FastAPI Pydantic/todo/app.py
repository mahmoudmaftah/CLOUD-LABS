from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Furniture model setup ---
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FURN_CSV = os.path.join(os.path.dirname(BASE_DIR), "furniture.csv")  # repo sibling file

furniture_model = None
label_encoders = {}
furn_mode_values = {}

def prepare_furniture_preprocessing():
    """Load furniture.csv and fit label encoders exactly like the notebook did so user-friendly inputs
    (strings) can be converted to the integer labels the saved model expects."""
    global label_encoders, furn_mode_values
    if not os.path.exists(FURN_CSV):
        # If the original CSV isn't found, we won't be able to map strings -> labels.
        return

    df = pd.read_csv(FURN_CSV, names=['item_id','name','category','old_price','sellable_online',
                                      'link','other_colors','short_description','designer','depth',
                                      'height','width','price'], skiprows=1, header=None)
    cols = ['category','sellable_online','other_colors']
    # Replace "?" with NaN and fill with mode as in notebook
    for c in df.columns:
        df[c] = df[c].replace("?", np.nan)
    for c in cols:
        df[c] = df[c].fillna(df[c].value_counts().index[0])
    # Fit simple label mapping dicts
    for c in cols:
        uniques = list(pd.Series(df[c].astype(str)).unique())
        mapping = {val: i for i, val in enumerate(uniques)}
        label_encoders[c] = mapping
        furn_mode_values[c] = df[c].mode().iloc[0]

def encode_furniture_input(col, value):
    """Encode a single furniture input value to integer label if mapping available."""
    if col in label_encoders:
        key = str(value)
        mapping = label_encoders[col]
        if key in mapping:
            return mapping[key]
        # unknown value: return mode's mapped value
        mode_val = str(furn_mode_values.get(col, list(mapping.keys())[0]))
        return mapping.get(mode_val, 0)
    # fallback: try numeric
    try:
        return float(value)
    except Exception:
        return 0

@app.on_event("startup")
def startup():
    global furniture_model
    # Load furniture model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            furniture_model = pickle.load(f)

    # Prepare encoders from CSV if available
    prepare_furniture_preprocessing()

    # Ensure wine artifacts exist (train if missing)
    wine_art_path = os.path.join(BASE_DIR, "wine_artifacts.joblib")
    if not os.path.exists(wine_art_path):
        _train_and_save_wine_artifacts(wine_art_path)

def _train_and_save_wine_artifacts(path):
    """Train a compact wine classifier and save artifacts (model, scaler, features, target_names)."""
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # basic preprocessing: fillna (dataset has none)
    X = X.fillna(X.median())

    # Use a RandomForest to get top features
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = list(importances.sort_values(ascending=False).head(5).index)

    # scale selected features and train a simple classifier
    scaler = StandardScaler()
    X_sel = X[top_features]
    X_scaled = scaler.fit_transform(X_sel)

    clf = LogisticRegression(max_iter=2000, random_state=0)
    clf.fit(X_scaled, y)

    artifacts = {
        'model': clf,
        'scaler': scaler,
        'features': top_features,
        'target_names': list(data.target_names)
    }
    joblib.dump(artifacts, path)

def load_wine_artifacts():
    path = os.path.join(BASE_DIR, "wine_artifacts.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/furniture", response_class=HTMLResponse)
def furniture_form(request: Request):
    # Provide placeholders or available categories if we have mappings
    categories = None
    if 'category' in label_encoders:
        categories = list(label_encoders['category'].keys())
    return templates.TemplateResponse("furniture_form.html", {"request": request, "categories": categories})


@app.post("/furniture/predict", response_class=HTMLResponse)
async def furniture_predict(request: Request,
                            category: str = Form(...),
                            sellable_online: str = Form(...),
                            other_colors: str = Form(...),
                            depth: float = Form(...),
                            height: float = Form(...),
                            width: float = Form(...)):
    if furniture_model is None:
        return templates.TemplateResponse("furniture_result.html", {"request": request, "error": "Model not found."})

    # encode inputs
    a = encode_furniture_input('category', category)
    b = encode_furniture_input('sellable_online', sellable_online)
    c = encode_furniture_input('other_colors', other_colors)
    try:
        d = float(depth)
        e = float(height)
        f = float(width)
    except Exception:
        return templates.TemplateResponse("furniture_result.html", {"request": request, "error": "Numeric feature parsing failed."})

    features = [a, b, c, d, e, f]
    try:
        pred = furniture_model.predict([features])[0]
    except Exception as ex:
        return templates.TemplateResponse("furniture_result.html", {"request": request, "error": str(ex)})

    return templates.TemplateResponse("furniture_result.html", {"request": request, "prediction": float(pred), "features": features})


@app.get("/wine", response_class=HTMLResponse)
def wine_form(request: Request):
    artifacts = load_wine_artifacts()
    if artifacts is None:
        return templates.TemplateResponse("wine_result.html", {"request": request, "error": "Wine artifacts not available."})
    features = artifacts['features']
    return templates.TemplateResponse("wine_form.html", {"request": request, "features": features})


@app.post("/wine/predict", response_class=HTMLResponse)
async def wine_predict(request: Request):
    artifacts = load_wine_artifacts()
    if artifacts is None:
        return templates.TemplateResponse("wine_result.html", {"request": request, "error": "Wine artifacts not available."})

    features = artifacts['features']
    values = []
    form = await request.form()
    for f in features:
        # form fields will be named exactly as feature names
        val = form.get(f)
        try:
            values.append(float(val))
        except Exception:
            return templates.TemplateResponse("wine_result.html", {"request": request, "error": f"Invalid value for {f}: {val}"})

    X = np.array(values).reshape(1, -1)
    X_scaled = artifacts['scaler'].transform(X)
    pred_idx = int(artifacts['model'].predict(X_scaled)[0])
    pred_name = artifacts['target_names'][pred_idx]

    return templates.TemplateResponse("wine_result.html", {"request": request, "prediction": pred_name, "features": dict(zip(features, values))})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
