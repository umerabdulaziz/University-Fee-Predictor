import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("fee_data.csv")

df["Net_Fee"] = (df["Tuition"] + df["Additional_Fees"]) * (1- df["Financial_Aid"]/100)

X = df[["Country", "Degree", "University_Type", "Field", "GPA"]]
y = df["Net_Fee"]

categorical = ["Country", "Degree", "University_Type", "Field"]
numerical = ["GPA"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numerical)
])

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  # ✅ Add this Input layer
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs= 500, verbose=0)

model.save("model.keras")
import joblib
joblib.dump(preprocessor, "preprocessor.pkl")

print("✅ Model and preprocessor saved.")