import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import joblib

def load_dataset(path="data/processed/train_final.csv"):
    df = pd.read_csv(path, low_memory=False)
    return df

def prepare_data(df):
    df = df.dropna(subset=['body'])  # drop rows with missing body
    drop_cols = ['subject', 'body', 'label']
    X = df.drop(columns=drop_cols)
    y = df['label']
    return X, y

def train_model():
    df = load_dataset()
    print("Dataset shape:", df.shape)

    X, y = prepare_data(df)
    print("After dropping NaN bodies:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)]
    )

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\n--- Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
    print("PR-AUC:", average_precision_score(y_test, y_pred_prob))

    joblib.dump(model, "data/processed/lgbm_model.pkl")
    print("Model saved.")

    return model, X_test, y_test, y_pred_prob

if __name__ == "__main__":
    train_model()