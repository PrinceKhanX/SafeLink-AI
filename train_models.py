import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Dictionary to store metrics for both experiments
results_full = {}
results_no_similarity = {}

print(f"\n{'='*50}")
print("EXPERIMENT 1: With URLSimilarityIndex")
print('='*50)

# Train and evaluate each model with all features
for name, model in models.items():
    print(f"\nTraining {name}")
    print('-'*50)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results_full[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Save model
    filename = name.lower().replace(' ', '_') + '.pkl'
    joblib.dump(model, f'models/{filename}')
    print(f"Model saved to models/{filename}")

# Print summary table for experiment 1
print(f"\n{'='*50}")
print("Summary Table - Experiment 1 (With URLSimilarityIndex)")
print('='*50)
summary_df_full = pd.DataFrame(results_full).T
print(summary_df_full)

# Experiment 2: Drop URLSimilarityIndex
print(f"\n{'='*50}")
print("EXPERIMENT 2: Without URLSimilarityIndex")
print('='*50)

X_train_no_sim = X_train.drop(columns=['URLSimilarityIndex'])
X_test_no_sim = X_test.drop(columns=['URLSimilarityIndex'])

# Reinitialize models for second experiment
models_2 = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Train and evaluate each model without URLSimilarityIndex
for name, model in models_2.items():
    print(f"\nTraining {name}")
    print('-'*50)
    
    # Train
    model.fit(X_train_no_sim, y_train)
    
    # Predict
    y_pred = model.predict(X_test_no_sim)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results_no_similarity[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Save model
    filename = name.lower().replace(' ', '_') + '_no_similarity.pkl'
    joblib.dump(model, f'models/{filename}')
    print(f"Model saved to models/{filename}")

# Print summary table for experiment 2
print(f"\n{'='*50}")
print("Summary Table - Experiment 2 (Without URLSimilarityIndex)")
print('='*50)
summary_df_no_sim = pd.DataFrame(results_no_similarity).T
print(summary_df_no_sim)
