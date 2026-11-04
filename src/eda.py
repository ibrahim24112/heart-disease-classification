import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def perform_eda(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(data_path)
    
    # 1. Target Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df, palette='viridis')
    plt.title('Distribution of Heart Disease (Target)')
    plt.xlabel('Target (0: No, 1-4: Yes)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 3. Age vs Target
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', hue='target', multiple='stack', palette='magma')
    plt.title('Age Distribution by Target')
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
    plt.close()
    
    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    data_path = str(Path(__file__).resolve().parents[1] / 'data' / 'heart_disease_clean.csv')
    output_dir = str(Path(__file__).resolve().parents[1] / 'output')
    perform_eda(data_path, output_dir)
