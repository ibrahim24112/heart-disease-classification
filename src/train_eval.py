import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import os
from pathlib import Path

def train_and_evaluate(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(data_path)
    
    # Pre-processing
    # Handle missing values (mean imputation for numeric)
    df = df.fillna(df.mean())
    
    # Binary classification: map target > 0 to 1
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv)
        
        # Fit and evaluate on test set
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "test_accuracy": float(acc),
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1-score": float(report["weighted avg"]["f1-score"])
        }
        
        # Plot Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, f'cm_{name.replace(" ", "_")}.png'))
        plt.close()

    # Save results to JSON
    with open(os.path.join(output_dir, 'model_comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Comparison Plot
    names = list(results.keys())
    accuracies = [results[n]["test_accuracy"] for n in names]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=accuracies, palette='Set2')
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    print(f"Training and evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    data_path = str(Path(__file__).resolve().parents[1] / 'data' / 'heart_disease_clean.csv')
    output_dir = str(Path(__file__).resolve().parents[1] / 'output')
    train_and_evaluate(data_path, output_dir)
