import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_evaluate(data_path='data/Clean_Dataset.csv'):
    print("Loading flight dataset...")
    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Select features and target
    features = ['airline', 'source_city', 'departure_time', 'stops',
                'arrival_time', 'destination_city', 'class', 'duration', 'days_left']
    X = pd.get_dummies(df[features], drop_first=True)
    y = df['price']

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Model Evaluation Metrics ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : ₹{mae:,.2f}")
    print(f"RMSE     : ₹{rmse:,.2f}")

    # Sample prediction on a test record
    sample_actual = y_test.iloc[0]
    sample_pred = y_pred[0]
    print(f"\nSample Test Flight -> Actual Fare: ₹{sample_actual:,.2f} | Predicted Fare: ₹{sample_pred:,.2f}")

if __name__ == "__main__":
    train_and_evaluate()
