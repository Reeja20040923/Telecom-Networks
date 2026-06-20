from app import app, load_data, load_models, load_preprocessors

# ── Called at startup BEFORE first request ────────────────────────────────────
print("=" * 55)
print("  Cellular Performance & Fault Diagnosis")
print("  Starting up...")
print("=" * 55)

# Load data
print("[1/3] Loading CSV data...")
load_data()

# Load models  ← THIS WAS THE BUG — it was never called before
print("[2/3] Loading ML models from models/ folder...")
load_models()

# Load preprocessors
print("[3/3] Loading scaler and label encoders...")
load_preprocessors()

from app import app_data
print(f"\n  Data loaded:   {'YES — ' + str(len(app_data['df'])) + ' rows' if app_data['df'] is not None else 'NO — place CSV in data/'}")
print(f"  Models loaded: {list(app_data['models'].keys()) if app_data['models'] else 'NONE — place .pkl files in models/'}")
print(f"\n  Open browser → http://localhost:5000")
print("=" * 55 + "\n")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
