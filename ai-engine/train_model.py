import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# 1. Génération de données d'entraînement simulées (Logs système / Trafic réseau)
def generate_mock_security_data(n_samples=2000):
    np.random.seed(42)
    # Features : [Failed Login Count, Request Rate (req/s), Payload Size (bytes), Session Duration (s)]
    normal_traffic = np.random.normal(loc=[1.0, 5.0, 500.0, 120.0], scale=[0.5, 1.5, 50.0, 20.0], size=(n_samples, 4))
    
    # Injection de comportements anormaux (ex: force brute, scans)
    anomalous_traffic = np.array([
        [15.0, 50.0, 120.0, 2.0],
        [0.0, 200.0, 30.0, 0.5],
        [25.0, 80.0, 1500.0, 5.0]
    ])
    
    data = np.vstack([normal_traffic, anomalous_traffic])
    return pd.DataFrame(data, columns=["failed_logins", "request_rate", "payload_size", "session_duration"])

def train_and_save_model():
    print("[*] Generating training dataset...")
    df = generate_mock_security_data(n_samples=2000)
    
    print("[*] Training Isolation Forest model for anomaly detection...")
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    model.fit(df)
    
    # Création du dossier de sortie pour le modèle
    os.makedirs("models", exist_ok=True)
    model_path = "models/isolation_forest_model.joblib"
    
    print(f"[*] Saving trained model to {model_path}...")
    joblib.dump(model, model_path)
    print("[+] Model training completed and saved successfully!")

if __name__ == "__main__":
    train_and_save_model()