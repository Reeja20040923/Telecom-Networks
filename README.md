# Telecom-Networks

# 📡 Fault Prediction in Telecom Networks

An intelligent **Machine Learning-based web application** that predicts network faults and diagnoses cellular network performance issues using real-time telecom signal metrics. The system provides predictive analytics, interactive dashboards, exploratory data analysis, and automated fault alerts to help telecom operators monitor and optimize network performance.

---

## 🚀 Features

### 📊 Exploratory Data Analysis (EDA)

* Network type distribution visualization
* Signal quality analysis across localities
* Throughput vs. latency correlation analysis
* Device-wise signal comparison charts
* Interactive graphical insights

### 🤖 Machine Learning-Based Prediction

* Predicts cellular network type (**3G, 4G, LTE, 5G**)
* Estimates signal strength using telecom metrics
* Supports multiple ML models:

  * Ridge Classifier
  * Decision Tree Classifier
  * Hybrid CatBoost Model

### 📍 Locality-Based Diagnosis

* Spatially aware predictions using:

  * Latitude
  * Longitude
  * Locality encoding
* Predicts network conditions for different geographical regions.

### 📈 Real-Time Monitoring Dashboard

* Live telecom signal simulation
* Real-time updates using Server-Sent Events (SSE)
* Continuous monitoring of:

  * Signal Strength
  * Signal Quality
  * Throughput
  * Latency

### 🚨 Automated Fault Detection & Alerts

Generates alerts for:

* Low signal strength
* High latency
* Low throughput
* Critical network failures

### 📂 Batch Prediction

* Upload CSV files for bulk predictions
* Download prediction results automatically

---

# 🛠️ Tech Stack

## Frontend

* HTML5
* CSS3
* JavaScript
* Bootstrap

## Backend

* Python
* Flask

## Machine Learning Libraries

* Scikit-learn
* Pandas
* NumPy
* Joblib

## Visualization Libraries

* Matplotlib

---

# 📂 Project Structure

```text
Fault prediction in telecom networks/
│
├── app.py                  # Main Flask application
├── main.py                 # Application entry point
│
├── data/                   # Training and test datasets
├── models/                 # Saved ML models (.pkl files)
├── results/                # Generated prediction outputs
│
├── templates/              # HTML templates
├── static/                 # CSS, JS, images, plots
│
└── __pycache__/
```

---

# 📊 Dataset Features

The model uses the following telecom parameters:

* Locality
* Latitude
* Longitude
* Signal Quality (%)
* Data Throughput (Mbps)
* Latency (ms)
* BB60C Measurement (dBm)
* srsRAN Measurement (dBm)
* BladeRFxA9 Measurement (dBm)

---

# 🧠 Machine Learning Workflow

### Data Collection

⬇️

### Data Preprocessing

* Missing value handling
* Label Encoding
* Feature Scaling

⬇️

### Model Training

* Ridge Classifier
* Decision Tree Classifier
* Hybrid CatBoost Model

⬇️

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* RMSE
* MAE
* R² Score

⬇️

### Prediction & Fault Diagnosis

---

# ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Fault-prediction-in-telecom-networks.git
cd Fault-prediction-in-telecom-networks
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/Mac**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install flask pandas numpy matplotlib scikit-learn joblib
```

---

# ▶️ Running the Application

### Start the Flask Server

```bash
python main.py
```

You should see:

```text
Cellular Performance & Fault Diagnosis
Starting up...

[1/3] Loading CSV data...
[2/3] Loading ML models...
[3/3] Loading scaler and label encoders...

Open browser → http://localhost:5000
```

Open:

```text
http://localhost:5000
```

---

# 📸 Application Modules

### 🏠 Home Page

Overview of datasets and loaded models.

### 📊 EDA Module

Interactive visual analysis of telecom metrics.

### 📈 Dashboard

Real-time monitoring and fault alerts.

### 🔍 Prediction Module

Predict network type and estimate signal strength.

### 📂 Batch Prediction

Upload CSV files and download prediction results.

---

# 🎯 Project Objectives

* Predict telecom network faults proactively.
* Improve cellular network reliability.
* Enable real-time performance monitoring.
* Support data-driven network optimization.
* Reduce downtime and service disruptions.

---

# 🔮 Future Enhancements

* Integration with real telecom APIs
* Deep Learning models for fault prediction
* Interactive geographical maps
* Cloud deployment (AWS/Azure)
* SMS and Email alert notifications
* Predictive maintenance recommendations

---

# 👩‍💻 Author

**Reeja Yaramalla**
B.Tech – Artificial Intelligence & Machine Learning
Passionate about Machine Learning, Data Analytics, and Intelligent Software Systems.

---

⭐ **If you found this project useful, please give it a star on GitHub!**
