# Autonomous ML Model Monitoring & Drift Management Platform 🚀

![SaaS Dashboard Preview](https://img.shields.io/badge/Status-Active-brightgreen) ![License](https://img.shields.io/badge/License-MIT-blue) ![Version](https://img.shields.io/badge/Version-1.0.0-orange)

An end-to-end, real-time machine learning monitoring solution designed to give data science and ML engineering teams deep visibility into their production models. Detect data drift, monitor model performance, and track anomalies automatically through a beautiful, modern SaaS interface.

---

## 🌟 Key Features

*   **Real-time Dashboard Analytics:** Monitor vital model metrics, predictions, and recent alerts dynamically at a glance.
*   **Statistical Drift Detection:** Automatically track Population Stability Index (PSI), Kolmogorov-Smirnov (KS) Statistic, and KL Divergence for all registered models.
*   **Model Registry & Management:** Upload trained model artifacts (`.pkl`, `.joblib`, `.h5`, etc.) securely and manage their versions seamlessly.
*   **Live Predictions & Probabilities:** Instantly execute predictions against your live endpoints with high-performance automated feature inference.
*   **Automated Alerting Engine:** Intelligent thresholds alert your team immediately when data drift or performance degradation is detected.

## 🏗 Architecture & Tech Stack

This platform is divided into a robust RESTful Python backend and a highly responsive Next.js frontend, orchestrated together for seamless deployment.

### Backend (API Engine)
*   **Framework:** Flask
*   **Database:** SQLite / SQLAlchemy (ORM)
*   **ML Integration:** Scikit-Learn, NumPy, Joblib
*   **Security:** JWT Authentication
*   **Deployment:** Gunicorn

### Frontend (User Interface)
*   **Framework:** Next.js (React)
*   **Styling:** Tailwind CSS (with custom SaaS UI components)
*   **Charts:** Recharts for dynamic, fluid data visualization
*   **Icons & Animation:** Lucide React, Framer Motion

---

## ⚙️ Local Setup Instructions

To run this platform locally for development and testing:

### 1. Backend Setup

Navigate to the `backend` directory, set up your Python environment, and start the Flask server:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python app.py
```
*The backend will be available at `http://localhost:5000`*

### 2. Frontend Setup

In a new terminal window, navigate to the `frontend` directory:

```bash
cd frontend
npm install
npm run dev
```
*The frontend will be available at `http://localhost:3000`*

---
