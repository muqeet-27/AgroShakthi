import pandas as pd
import joblib   # 🔹 Add this
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv("/kaggle/input/githubdata/Crops_data.csv")   # change file name

# =========================
# 2. Define Function for Training a Crop Model
# =========================
def train_model_for_crop(target_crop):
    target_col = f"{target_crop.upper()} YIELD (Kg per ha)"

    if target_col not in data.columns:
        raise ValueError(f"❌ {target_col} not found in dataset")

    # Drop other yield columns
    yield_cols = [c for c in data.columns if "YIELD" in c and c != target_col]

    categorical_cols = ["State Name", "Dist Name"]
    numerical_cols = [col for col in data.columns
                      if col not in categorical_cols + yield_cols + [target_col]]

    X = data[categorical_cols + numerical_cols]
    y = data[target_col]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )

    # Model pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [10, 20, None]
    }

    grid = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    print(f"\n✅ {target_crop} Model Performance:")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))

    return best_model, X

# =========================
# 3. Train for a Chosen Crop
# =========================
chosen_crop = "MAIZE"
model, X_features = train_model_for_crop(chosen_crop)

# =========================
# 4. User Input Example
# =========================
user_input = {
    "State Name": "Karnataka",
    "Dist Name": "Davangere",
    "Dist Code": 123,
    "Year": 2021,
    "State Code": 29,
    "RICE AREA (1000 ha)": 150,
    "RICE PRODUCTION (1000 tons)": 400,
    "WHEAT AREA (1000 ha)": 10,
    "WHEAT PRODUCTION (1000 tons)": 20,
    "MAIZE AREA (1000 ha)": 50,
    "MAIZE PRODUCTION (1000 tons)": 120,
    # Add other crop stats...
}

# Convert to DataFrame
user_df = pd.DataFrame([user_input])
user_df = user_df.reindex(columns=X_features.columns, fill_value=0)

# Prediction
prediction = model.predict(user_df)[0]
print(f"\n🌾 Predicted {chosen_crop} Yield (Kg per ha):", round(prediction, 2))

# =========================
# 5. Save model
# =========================
joblib.dump(model, "/kaggle/working/crop_yield_model.pkl")
print("✅ Model saved at /kaggle/working/crop_yield_model.pkl")
