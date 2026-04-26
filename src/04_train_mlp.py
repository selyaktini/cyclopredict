import pandas as pd
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

# files paths
DATA_PATH = "./data/processed/final_training_set.csv"
MODEL_DIR = "./models/"

def train_model():
    # load data
    df = pd.read_csv(DATA_PATH)
    
    # separe Features (X) and Target (y)
    X = df.drop(columns=['comptage_1h'])
    y = df['comptage_1h']

    # Split data (80% Train / 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # standarisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # init MLP  
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )

    # train
    print("start training MLP...")
    mlp.fit(X_train_scaled, y_train)

    # eval
    predictions = mlp.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    print(f"finish eval.\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}")

    # save artefacts
    joblib.dump(mlp, os.path.join(MODEL_DIR, "mlp_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print(f"model and scalar saved in {MODEL_DIR}")

if __name__ == "__main__":
    train_model()
