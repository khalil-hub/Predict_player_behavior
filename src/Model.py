import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Define directories
logs_dir = "results/logs"
predictions_dir = "results/prediction"

# Ensure directories exist
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# Load datasets
X_train = pd.read_csv('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Predict online behavior kaggle/Data/processed/x_train_online_gaming_behavior_dataset.csv')
X_test = pd.read_csv('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Predict online behavior kaggle/Data/processed/x_test_online_gaming_behavior_dataset.csv')
y_train = pd.read_csv('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Predict online behavior kaggle/Data/processed/y_train_online_gaming_behavior_dataset.csv')
y_test = pd.read_csv('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Predict online behavior kaggle/Data/processed/y_test_online_gaming_behavior_dataset.csv')

# Ensure y_train and y_test are 1D
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Initialization
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1, max_depth=10),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "LightGBM": LGBMClassifier(verbose=-1, random_state=42, class_weight='balanced'),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

# Initialize results storage
model_results = []

# Train and evaluate models
for model_name, model in models.items():
    print(f"\nTraining and evaluating: {model_name}")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Save the model
    model_path = os.path.join(logs_dir, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Check if the model supports predict_proba
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Ensure correct shape for multiclass classification
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:  # Binary classification case
            auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = "N/A"
    
    print(f"y_pred_proba shape: {y_pred_proba.shape if y_pred_proba is not None else 'N/A'}")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    model_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc
    })
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Create a summary table of results
results_df = pd.DataFrame(model_results).sort_values(by="Accuracy", ascending=False)
results_df.reset_index(drop=True, inplace=True)

# Save results to CSV
results_path = os.path.join(predictions_dir, "model_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to {results_path}")

# Display the summary results
print("\nSummary of Model Evaluation:")
print(results_df)