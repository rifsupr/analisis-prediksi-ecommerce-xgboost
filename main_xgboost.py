# =====================================================
# 1. IMPORT LIBRARY
# =====================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
# =====================================================
# 2. LOAD DATA
# =====================================================
def load_data(path):
    df = pd.read_csv(path)
    print("------------------------------------------")
    print("1) INFORMASI DATA:")
    print("------------------------------------------")
    print(f"Jumlah Record   : {df.shape[0]}")
    print(f"Jumlah Variabel : {df.shape[1]}")

    return df
# =====================================================
# 3. CEK MISSING VALUE
# =====================================================
def check_missing(df, title="Missing Value"):
    print("------------------------------------------")
    print(f"{title} ")
    print("------------------------------------------")
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        "Missing": missing_count,
        "(%)": missing_percent.round(2)
    })

    print(missing_df)    
# =====================================================
# 4. SIMULASI MISSING VALUE
# =====================================================
def simulate_missing(df, cols, frac=0.05):
    df_copy = df.copy()
    np.random.seed(42)

    for col in cols:
        df_copy.loc[df_copy.sample(frac=frac).index, col] = np.nan

    return df_copy
# =====================================================
# 5. IMPUTASI
# =====================================================
def impute_data(df):
    df_copy = df.copy()
    # -------------------------
    # Tipe data Numerik menggunakan median
    # -------------------------
    num_cols = ["price", "freight_value", "product_weight_g"]
    for col in num_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    # -------------------------
    # Tipe data Kategorikal menggunakan modus
    # -------------------------
    cat_cols = ["product_category_name", "customer_state", "payment_type"]
    for col in cat_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

    return df_copy
# =====================================================
# 6. TRANSFORMASI DATA
# =====================================================
def transform_data(df):
    df_copy = df.copy()

    X = df_copy.drop(columns=["order_id", "payment_value"])
    y = df_copy["payment_value"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    # -------------------------
    # Scaling
    # -------------------------
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    # -------------------------
    # Encoding
    # -------------------------
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print("------------------------------------------")
    print("Transformasi selesai. Shape:", X.shape)
    print("------------------------------------------")

    return X, y
# =====================================================
# 7. SPLIT DATA
# =====================================================
def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)
# =====================================================
# 8. TRAIN MODEL
# =====================================================
def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    return model
# =====================================================
# 9. EVALUASI MODEL
# =====================================================
def evaluate_model(model, X_test, y_test, label=""):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("------------------------------------------")
    print(f" HASIL SPLIT {label} ")
    print("------------------------------------------")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)
    print("------------------------------------------")

    return y_pred
# =====================================================
# 10. VISUALISASI
# =====================================================
def plot_distribution(y):
    plt.figure()
    sns.histplot(y, kde=True)
    plt.title("Distribusi Payment Value")
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.select_dtypes(include=["float64","int64"]).corr(),
                annot=True, cmap="coolwarm")
    plt.title("CORRELATION MATRIX")
    plt.show()

def plot_prediction(y_true, y_pred, title):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()

def plot_feature_importance(model):
    plt.figure(figsize=(10,6))
    xgb.plot_importance(model, max_num_features=10)
    plt.title("Feature Importance")
    plt.show()
# =====================================================
# 11. MAIN PIPELINE
# =====================================================
def run_pipeline():
    # -------------------------
    # Load
    # -------------------------
    df = load_data("datasets/olist_orders_success.csv")
    # -------------------------
    # Missing check
    # -------------------------
    check_missing(df, "2. SEBELUM PENANGANAN MISSING VALUE")
    # -------------------------
    # Simulasi
    # -------------------------
    df = simulate_missing(df, ["price", "freight_value", "product_weight_g"])
    check_missing(df, "3. SIMULASI MISSING VALUE > 5%")
    # -------------------------
    # Imputasi
    # -------------------------
    df = impute_data(df)
    check_missing(df, "2. SETELAH PENANGANAN MISSING VALUE")
    # -------------------------
    # Visual awal
    # -------------------------
    plot_distribution(df["payment_value"])
    plot_correlation(df)
    # -------------------------
    # Transformasi
    # -------------------------
    X, y = transform_data(df)
    # -------------------------
    # SKENARIO 80:20
    # -------------------------
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    model_80 = train_xgboost(X_train, y_train)
    y_pred_80 = evaluate_model(model_80, X_test, y_test, "80:20")

    plot_prediction(y_test, y_pred_80, "ACTUAL vs PREDICTED (80:20)")
    # -------------------------
    # SKENARIO 70:30
    # -------------------------
    X_train, X_test, y_train, y_test = split_data(X, y, 0.3)

    model_70 = train_xgboost(X_train, y_train)
    y_pred_70 = evaluate_model(model_70, X_test, y_test, "70:30")

    plot_prediction(y_test, y_pred_70, "ACTUAL vs PREDICTED (70:30)")
    # -------------------------
    # Feature importance
    # -------------------------
    plot_feature_importance(model_80)
# =====================================================
# 12. RUN
# =====================================================
if __name__ == "__main__":
    run_pipeline()