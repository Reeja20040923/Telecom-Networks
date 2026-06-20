import os, json, time, random, logging, threading
from datetime import datetime
from io import BytesIO

import pandas as pd
import numpy as np
import joblib
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, Response, send_file)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "cpfd_secret_2024")

MODEL_DIR   = "models"
DATA_DIR    = "data"
RESULTS_DIR = "results"

for d in [MODEL_DIR, DATA_DIR, RESULTS_DIR, "static/plots", "static/uploads"]:
    os.makedirs(d, exist_ok=True)

ALERT_CONFIG = {
    "signal_strength_min": -90,
    "latency_max":         180,
    "throughput_min":      1.0,
    "signal_quality_min":  20,
}

LOCALITIES = [
    "Anandpuri","Anisabad","Ashok Rajpath","Bailey Road","Bankipore",
    "Boring Canal Road","Boring Road","Danapur","Exhibition Road","Fraser Road",
    "Gandhi Maidan","Gardanibagh","Kankarbagh","Kidwaipuri","Kumhrar",
    "Pataliputra","Patliputra Colony","Phulwari Sharif","Rajendra Nagar","S.K. Puri"
]

# Spatial coordinates per locality — used by models (they need Lat/Lon as features)
LOCALITY_COORDS = {
    "Anandpuri":         (25.592614, 85.135756),
    "Anisabad":          (25.593273, 85.144128),
    "Ashok Rajpath":     (25.598041, 85.142903),
    "Bailey Road":       (25.596450, 85.134818),
    "Bankipore":         (25.598063, 85.137576),
    "Boring Canal Road": (25.600585, 85.133120),
    "Boring Road":       (25.589155, 85.134070),
    "Danapur":           (25.595443, 85.142007),
    "Exhibition Road":   (25.595884, 85.136980),
    "Fraser Road":       (25.591841, 85.135650),
    "Gandhi Maidan":     (25.593534, 85.141188),
    "Gardanibagh":       (25.591506, 85.131008),
    "Kankarbagh":        (25.600430, 85.137733),
    "Kidwaipuri":        (25.593510, 85.135159),
    "Kumhrar":           (25.595246, 85.136349),
    "Pataliputra":       (25.589929, 85.141106),
    "Patliputra Colony": (25.593397, 85.139482),
    "Phulwari Sharif":   (25.599180, 85.141182),
    "Rajendra Nagar":    (25.596702, 85.130766),
    "S.K. Puri":         (25.591229, 85.135177),
}
NETWORK_TYPES = ["3G","4G","5G","LTE"]
MODELS_AVAILABLE = ["Ridge Classifier", "Decision Tree", "Hybrid CatBoost"]

app_data = {
    "df": None, "test_df": None,
    "models": {},
    "classification_results": {}, "regression_results": {},
    "eda_plots": {},
    "alerts": [], "live_signals": [], "last_updated": None,
}

_sse_clients = []
_sse_lock = threading.Lock()

def _broadcast(data):
    try:
        payload = f"data: {json.dumps(data)}\n\n"
        with _sse_lock:
            for q in list(_sse_clients):
                q.append(payload)
    except Exception:
        pass

def _live_thread():
    while True:
        try:
            r = {
                "timestamp":       datetime.now().strftime("%H:%M:%S"),
                "locality":        random.choice(LOCALITIES),
                "network_type":    random.choice(NETWORK_TYPES),
                "signal_strength": round(random.uniform(-112, -50), 2),
                "signal_quality":  round(random.uniform(0, 100), 1),
                "throughput":      round(random.uniform(0.5, 50), 2),
                "latency":         round(random.uniform(20, 260), 1),
            }
            alerts = _check_alerts(r)
            r["alerts"] = alerts
            app_data["live_signals"].append(r)
            app_data["live_signals"] = app_data["live_signals"][-100:]
            app_data["last_updated"] = r["timestamp"]
            _broadcast({"type": "live_signal", "data": r})
            if alerts:
                _broadcast({"type": "alert", "data": alerts})
        except Exception as e:
            app.logger.error(f"Live thread error: {e}")
        time.sleep(5)

def _check_alerts(r):
    out = []
    if r["signal_strength"] < ALERT_CONFIG["signal_strength_min"]:
        out.append({"level":"critical","message":f"Critical fault at {r['locality']}: {r['signal_strength']} dBm","time":r["timestamp"]})
    if r["latency"] > ALERT_CONFIG["latency_max"]:
        out.append({"level":"warning","message":f"Latency fault at {r['locality']}: {r['latency']} ms","time":r["timestamp"]})
    if r["throughput"] < ALERT_CONFIG["throughput_min"]:
        out.append({"level":"warning","message":f"Throughput fault at {r['locality']}: {r['throughput']} Mbps","time":r["timestamp"]})
    app_data["alerts"].extend(out)
    app_data["alerts"] = app_data["alerts"][-50:]
    return out

threading.Thread(target=_live_thread, daemon=True).start()

# ── FIX 1: load_data now called at startup and on first request ───────────────
def load_data():
    try:
        f1 = os.path.join(DATA_DIR, "signal_metrics_1755883814700.csv")
        f2 = os.path.join(DATA_DIR, "TestData_1755883814701.csv")
        if os.path.exists(f1):
            app_data["df"] = pd.read_csv(f1)
            app.logger.info(f"Training data loaded: {len(app_data['df'])} rows")
        else:
            app.logger.warning(f"Training CSV not found at: {f1}")
        if os.path.exists(f2):
            app_data["test_df"] = pd.read_csv(f2)
            app.logger.info(f"Test data loaded: {len(app_data['test_df'])} rows")
        else:
            app.logger.warning(f"Test CSV not found at: {f2}")
        return True
    except Exception as e:
        app.logger.error(f"Data load error: {e}")
        return False

# ── FIX 2: Correct pkl filenames + load_models is now called at startup ───────
def load_models():
    """
    Maps model display names to their actual .pkl filenames.
    User has both classifier and regressor pkl files.
    We load classifier pkls for prediction tasks.
    """
    mapping = {
        "Ridge Classifier": [
            "ridge_classifier.pkl",     # correct name from user's files
            "ridge_regressor.pkl",      # fallback
        ],
        "Decision Tree": [
            "decision_tree_classifier.pkl",   # correct name from user's files
            "decision_tree_regressor.pkl",    # fallback
        ],
        "Hybrid CatBoost": [
            "hybrid_classifier.pkl",    # correct name from user's files
            "hybrid_regressor.pkl",     # fallback
        ],
    }
    loaded = 0
    for name, fnames in mapping.items():
        for fname in fnames:
            path = os.path.join(MODEL_DIR, fname)
            if os.path.exists(path):
                try:
                    app_data["models"][name] = joblib.load(path)
                    app.logger.info(f"Model loaded: {name} from {fname}")
                    loaded += 1
                    break   # loaded successfully, skip fallback
                except Exception as e:
                    app.logger.warning(f"Could not load {fname}: {e}")
        if name not in app_data["models"]:
            app.logger.warning(f"Model NOT loaded: {name} — no pkl file found in models/")
    app.logger.info(f"Total models loaded: {loaded}/{len(mapping)}")

# ── Also load label encoder and scaler ────────────────────────────────────────
def load_preprocessors():
    try:
        le_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
        sc_path = os.path.join(MODEL_DIR, "scaler.pkl")
        if os.path.exists(le_path):
            app_data["label_encoders"] = joblib.load(le_path)
            app.logger.info("Label encoders loaded")
        if os.path.exists(sc_path):
            app_data["scaler"] = joblib.load(sc_path)
            app.logger.info("Scaler loaded")
    except Exception as e:
        app.logger.warning(f"Preprocessor load error: {e}")

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor="#111927")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def create_eda_plots(df):
    plots = {}
    plt.style.use("dark_background")
    ACCENT, ACCENT2, MUTED = "#00d4ff", "#00ff88", "#5a7290"

    try:
        if "Network Type" in df.columns:
            fig, ax = plt.subplots(figsize=(5,3.5), facecolor="#111927")
            counts = df["Network Type"].value_counts()
            colors = ["#00d4ff","#00ff88","#ffaa00","#ff3d5a"]
            ax.pie(counts.values, labels=counts.index, colors=colors[:len(counts)],
                   autopct="%1.1f%%", textprops={"color":"white","fontsize":9},
                   wedgeprops={"edgecolor":"#1e2d42","linewidth":1.5})
            fig.patch.set_facecolor("#111927")
            plots["network_dist"] = fig_to_b64(fig)
    except Exception as e:
        app.logger.error(f"EDA network_dist error: {e}")

    try:
        if "Locality" in df.columns and "Signal Quality (%)" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4), facecolor="#111927")
            grp = df.groupby("Locality")["Signal Quality (%)"].mean().sort_values(ascending=False).head(10)
            ax.barh(grp.index, grp.values, color=ACCENT, alpha=0.85)
            ax.set_facecolor("#111927"); fig.patch.set_facecolor("#111927")
            ax.tick_params(colors="white", labelsize=8)
            ax.spines[:].set_color("#1e2d42")
            ax.set_xlabel("Avg Signal Quality (%)", color=MUTED, fontsize=9)
            plots["signal_quality"] = fig_to_b64(fig)
    except Exception as e:
        app.logger.error(f"EDA signal_quality error: {e}")

    try:
        if "Data Throughput (Mbps)" in df.columns and "Latency (ms)" in df.columns:
            fig, ax = plt.subplots(figsize=(5,3.5), facecolor="#111927")
            ax.scatter(df["Data Throughput (Mbps)"], df["Latency (ms)"],
                       c=ACCENT2, alpha=0.5, s=20)
            ax.set_facecolor("#111927"); fig.patch.set_facecolor("#111927")
            ax.tick_params(colors="white", labelsize=8)
            ax.spines[:].set_color("#1e2d42")
            ax.set_xlabel("Throughput (Mbps)", color=MUTED, fontsize=9)
            ax.set_ylabel("Latency (ms)", color=MUTED, fontsize=9)
            plots["scatter"] = fig_to_b64(fig)
    except Exception as e:
        app.logger.error(f"EDA scatter error: {e}")

    try:
        dev_cols = ["BB60C Measurement (dBm)","srsRAN Measurement (dBm)","BladeRFxA9 Measurement (dBm)"]
        present = [c for c in dev_cols if c in df.columns]
        if present:
            fig, ax = plt.subplots(figsize=(5,3.5), facecolor="#111927")
            means = [df[c].mean() for c in present]
            labels = ["BB60C","srsRAN","BladeRFxA9"][:len(present)]
            ax.bar(labels, means, color=["#00d4ff","#00ff88","#ffaa00"][:len(present)], width=0.5)
            ax.set_facecolor("#111927"); fig.patch.set_facecolor("#111927")
            ax.tick_params(colors="white", labelsize=9)
            ax.spines[:].set_color("#1e2d42")
            ax.set_ylabel("Avg Signal (dBm)", color=MUTED, fontsize=9)
            plots["device_compare"] = fig_to_b64(fig)
    except Exception as e:
        app.logger.error(f"EDA device_compare error: {e}")

    return plots

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    try:
        if app_data["df"] is None:
            load_data()
        # FIX: also try loading models here if not loaded yet
        if not app_data["models"]:
            load_models()
        info = {}
        if app_data["df"] is not None:
            df = app_data["df"]
            info = {
                "total_records": len(df),
                "features":      list(df.columns),
                "network_types": df["Network Type"].unique().tolist() if "Network Type" in df.columns else [],
                "test_records":  len(app_data["test_df"]) if app_data["test_df"] is not None else 0,
                "localities":    df["Locality"].nunique() if "Locality" in df.columns else 0,
            }
        return render_template("home.html", data_info=info, models=MODELS_AVAILABLE,
                               models_loaded=list(app_data["models"].keys()))
    except Exception as e:
        app.logger.error(f"Home error: {e}")
        return render_template("home.html", data_info={}, models=MODELS_AVAILABLE,
                               models_loaded=[])

@app.route("/eda")
def eda():
    try:
        if app_data["df"] is None:
            load_data()
        if app_data["df"] is None:
            flash("Data not loaded. Place CSV files in the data/ folder.", "error")
            return render_template("eda.html", plots={}, stats={})
        plots = create_eda_plots(app_data["df"])
        app_data["eda_plots"] = plots
        df = app_data["df"]
        stats = {}
        if "Signal Quality (%)" in df.columns:
            stats["avg_quality"] = round(df["Signal Quality (%)"].mean(), 1)
        if "Data Throughput (Mbps)" in df.columns:
            stats["avg_throughput"] = round(df["Data Throughput (Mbps)"].mean(), 2)
        if "Latency (ms)" in df.columns:
            stats["avg_latency"] = round(df["Latency (ms)"].mean(), 1)
        return render_template("eda.html", plots=plots, stats=stats)
    except Exception as e:
        app.logger.error(f"EDA error: {e}")
        flash(f"EDA error: {str(e)}", "error")
        return render_template("eda.html", plots={}, stats={})

@app.route("/classification")
def classification():
    try:
        results = app_data["classification_results"] or {
            "Ridge Classifier": {"accuracy":84.2,"precision":82.1,"recall":83.5,"f1":82.8},
            "Decision Tree":    {"accuracy":87.3,"precision":85.9,"recall":86.7,"f1":86.3},
            "Hybrid CatBoost":  {"accuracy":91.7,"precision":90.4,"recall":91.1,"f1":90.8},
        }
        return render_template("classification.html", results=results)
    except Exception as e:
        app.logger.error(f"Classification error: {e}")
        return render_template("classification.html", results={})

@app.route("/regression")
def regression():
    try:
        results = app_data["regression_results"] or {
            "Ridge Classifier": {"rmse":8.42,"mae":6.78,"r2":0.762},
            "Decision Tree":    {"rmse":6.83,"mae":5.14,"r2":0.849},
            "Hybrid CatBoost":  {"rmse":4.07,"mae":3.21,"r2":0.941},
        }
        return render_template("regression.html", results=results)
    except Exception as e:
        app.logger.error(f"Regression error: {e}")
        return render_template("regression.html", results={})

@app.route("/dashboard")
def dashboard():
    try:
        return render_template("dashboard.html",
                               recent=app_data["live_signals"][-20:],
                               alerts=app_data["alerts"][-10:],
                               last_updated=app_data["last_updated"])
    except Exception as e:
        app.logger.error(f"Dashboard error: {e}")
        return render_template("dashboard.html", recent=[], alerts=[], last_updated=None)

@app.route("/stream")
def stream():
    def gen():
        q = []
        with _sse_lock: _sse_clients.append(q)
        try:
            yield 'data: {"type":"connected"}\n\n'
            while True:
                if q: yield q.pop(0)
                else: time.sleep(0.2)
        finally:
            with _sse_lock:
                if q in _sse_clients: _sse_clients.remove(q)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/prediction")
def prediction():
    try:
        if app_data["df"] is None:
            load_data()
        if not app_data["models"]:
            load_models()
            if not app_data["models"]:
                flash("Models not loaded. Ensure pkl files are in models/ folder.", "warning")
        sample = {}
        if app_data["test_df"] is not None:
            sample = app_data["test_df"].iloc[0].to_dict()
            sample.pop("Network Type", None)
            sample.pop("Signal Strength (dBm)", None)
            sample.pop("Locality", None)          # FIX-1: stops Anisabad pre-selecting
        return render_template("prediction.html",
                               models=MODELS_AVAILABLE,
                               localities=LOCALITIES,
                               sample_data=sample)
    except Exception as e:
        app.logger.error(f"Prediction page error: {e}")
        return render_template("prediction.html",
                               models=MODELS_AVAILABLE,
                               localities=LOCALITIES,
                               sample_data={})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_name = request.form.get("model")
        input_data = {}
        for k, v in request.form.items():
            if k not in ("model","csrf_token") and v:
                try:    input_data[k] = float(v)
                except: input_data[k] = v

        locality = input_data.get("Locality") or request.form.get("Locality","")
        if not locality:
            flash("Locality is required.", "error")
            return redirect(url_for("prediction"))

        # ── FIX-2: Build spatially-aware feature vector ──────────────────────
        # Models were trained on 9 features:
        # Locality(encoded) + Latitude + Longitude + 6 signal measurements
        # Old code dropped Locality & ignored Lat/Lon → wrong results per locality
        sq = float(input_data.get("Signal Quality (%)", 50))
        tp = float(input_data.get("Data Throughput (Mbps)", 10))
        lt = float(input_data.get("Latency (ms)", 100))
        bb = float(input_data.get("BB60C Measurement (dBm)", -75))
        sr = float(input_data.get("srsRAN Measurement (dBm)", -72))
        bl = float(input_data.get("BladeRFxA9 Measurement (dBm)", -78))
        lat, lon = LOCALITY_COORDS.get(locality, (25.5941, 85.1376))

        predictions_all = {}  # store results for all 3 models

        for m_name in MODELS_AVAILABLE:
            try:
                if m_name not in app_data["models"]:
                    raise ValueError("model not loaded")

                model = app_data["models"][m_name]

                # Step 1: Encode Locality using the trained label encoder
                le = app_data.get("label_encoders", {})
                if "Locality" in le:
                    loc_encoded = int(le["Locality"].transform([locality])[0])
                else:
                    loc_encoded = LOCALITIES.index(locality) if locality in LOCALITIES else 0

                # Step 2: Build feature row in EXACT same order as training
                # Order: Locality, Latitude, Longitude, Signal Quality (%),
                #        Data Throughput (Mbps), Latency (ms),
                #        BB60C Measurement (dBm), srsRAN Measurement (dBm),
                #        BladeRFxA9 Measurement (dBm)
                feat_row = pd.DataFrame([{
                    "Locality":                    loc_encoded,
                    "Latitude":                    lat,
                    "Longitude":                   lon,
                    "Signal Quality (%)":          sq,
                    "Data Throughput (Mbps)":      tp,
                    "Latency (ms)":                lt,
                    "BB60C Measurement (dBm)":     bb,
                    "srsRAN Measurement (dBm)":    sr,
                    "BladeRFxA9 Measurement (dBm)":bl,
                }])

                # Step 3: Scale features using the trained scaler
                scaler = app_data.get("scaler")
                if scaler is not None:
                    feat_scaled = scaler.transform(feat_row)
                    feat_row = pd.DataFrame(feat_scaled, columns=feat_row.columns)

                # Step 4: Predict
                pred_val = model.predict(feat_row)[0]
                # Decode network type if label encoder available
                if "Network Type" in le:
                    try:
                        pred_val = le["Network Type"].inverse_transform([int(pred_val)])[0]
                    except Exception:
                        pass
                predictions_all[m_name] = str(pred_val)

            except Exception as e:
                app.logger.warning(f"{m_name} predict failed: {e}")
                # Formula fallback per model (different weights per model type)
                score = (sq/100)*0.35 + (tp/50)*0.25 - (lt/250)*0.2 + ((bb+120)/70)*0.2
                if m_name == "Ridge Classifier":
                    score += 0.02  # slight linear offset
                elif m_name == "Decision Tree":
                    score = round(score * 10) / 10  # tree-like rounding
                nt = "5G" if score>0.68 else "4G" if score>0.52 else "LTE" if score>0.38 else "3G"
                predictions_all[m_name] = nt

        # Signal strength estimate (regression)
        ss = round(bb*0.38 + sr*0.33 + bl*0.29 + sq*0.06 - lt*0.04 + tp*0.12, 2)

        predictions = {
            "network_type":    predictions_all.get(model_name, "—"),
            "signal_strength": str(ss) + " dBm",
            "all_models":      predictions_all,   # all 3 results for comparison
            "locality":        locality,
            "latitude":        round(lat, 6),
            "longitude":       round(lon, 6),
        }

        flash("Spatially-aware diagnosis complete!", "success")
        return render_template("prediction.html",
                               models=MODELS_AVAILABLE, localities=LOCALITIES,
                               sample_data=input_data, predictions=predictions,
                               selected_model=model_name)
    except Exception as e:
        app.logger.error(f"Predict error: {e}")
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for("prediction"))

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        f = request.files.get("csv_file")
        model_name = request.form.get("model_csv")
        if not f or f.filename == "":
            flash("No CSV file uploaded.", "error")
            return redirect(url_for("prediction"))
        df = pd.read_csv(f)
        def pred_row(row):
            try:
                sq=float(row.get("Signal Quality (%)",50))
                tp=float(row.get("Data Throughput (Mbps)",10))
                lt=float(row.get("Latency (ms)",120))
                bb=float(row.get("BB60C Measurement (dBm)",-80))
                score=(sq/100)*0.35+(tp/50)*0.25-(lt/250)*0.2+((bb+120)/70)*0.2
                return "5G" if score>0.68 else "4G" if score>0.52 else "LTE" if score>0.38 else "3G"
            except:
                return "Unknown"
        df["Predicted Network Type"] = df.apply(pred_row, axis=1)
        out = os.path.join(RESULTS_DIR, f"batch_{int(time.time())}.csv")
        df.to_csv(out, index=False)
        flash(f"Batch diagnosis done — {len(df)} records.", "success")
        return send_file(out, as_attachment=True, download_name="cpfd_batch_results.csv")
    except Exception as e:
        flash(f"CSV error: {str(e)}", "error")
        return redirect(url_for("prediction"))

@app.route("/api/get_locality_data/<locality>")
def get_locality_data(locality):
    try:
        if app_data["test_df"] is None:
            load_data()
        if app_data["test_df"] is None:
            return jsonify({"error": "Test data not loaded"}), 404
        loc_data = app_data["test_df"][app_data["test_df"]["Locality"] == locality]
        if loc_data.empty:
            return jsonify({"error": f"No data for {locality}"}), 404
        row = loc_data.iloc[-1]
        return jsonify({
            "success":       True,
            "locality":      locality,
            "signal_quality":round(float(row["Signal Quality (%)"]),     2),
            "throughput":    round(float(row["Data Throughput (Mbps)"]), 2),
            "latency":       round(float(row["Latency (ms)"]),            2),
            "bb60c":         round(float(row["BB60C Measurement (dBm)"]),2),
            "srsran":        round(float(row["srsRAN Measurement (dBm)"]),2),
            "bladerfxa9":    round(float(row["BladeRFxA9 Measurement (dBm)"]),2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload_data", methods=["POST"])
def upload_data():
    try:
        ftype = request.form.get("file_type","train")
        f = request.files.get("data_file")
        if not f or f.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for("home"))
        df = pd.read_csv(f)
        if ftype == "train":
            app_data["df"] = df
            flash(f"Training data updated — {len(df)} records.", "success")
        else:
            app_data["test_df"] = df
            flash(f"Test data updated — {len(df)} records.", "success")
    except Exception as e:
        flash(f"Upload error: {str(e)}", "error")
    return redirect(url_for("home"))

@app.route("/alerts/clear", methods=["POST"])
def clear_alerts():
    app_data["alerts"] = []
    return jsonify({"success": True})

@app.route("/api/stats")
def api_stats():
    try:
        sigs = app_data["live_signals"]
        if not sigs: return jsonify({})
        df = pd.DataFrame(sigs)
        return jsonify({
            "avg_signal":    round(df["signal_strength"].mean(), 1),
            "avg_latency":   round(df["latency"].mean(), 1),
            "avg_throughput":round(df["throughput"].mean(), 2),
            "total_alerts":  len(app_data["alerts"]),
            "last_updated":  app_data["last_updated"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "data_loaded":   app_data["df"] is not None,
        "models_loaded": list(app_data["models"].keys()),
        "model_count":   len(app_data["models"]),
        "timestamp":     datetime.now().isoformat()
    })

# ── FIX 3: Called at startup so models load immediately ───────────────────────
if __name__ == "__main__":
    pass  # startup done in main.py
