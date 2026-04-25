import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# Config
DATA_PATH = "./data/processed/final_training_set.csv"
MODEL_PATH = "./models/mlp_model.joblib"
SCALER_PATH = "./models/scaler.joblib"

def plot_results():
    # load
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # prepare set test 
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    X_test = test_df.drop(columns=['comptage_1h'])
    y_true = test_df['comptage_1h'].values

    # Inference
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # Visualisation (Zoom in 7 jours / 168 heures)
    plt.figure(figsize=(15, 6))
    plt.plot(y_true[:168], label="Réel (Comptage)", color='blue', alpha=0.7)
    plt.plot(y_pred[:168], label="Prédit (MLP)", color='red', linestyle='--')
    
    plt.title("Comparaison Flux Réel vs Prédit sur 7 jours (Axe Bordeaux-Campus)")
    plt.xlabel("Hours")
    plt.ylabel("Nombre de vélos")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.savefig("./plots/results_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_results()
