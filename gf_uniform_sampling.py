import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset

# 1. Load and Prepare Data
def load_data():
    import kagglehub
    path = kagglehub.dataset_download("uciml/adult-census-income")
    df = pd.read_csv(path + '/adult.csv')
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    # Convert target and protected attribute
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})
    
    # Convert other categorical columns to numerical codes
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'native-country']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    return df.dropna()

# 2. Uniform Sampling Preprocessing
def uniform_sampling(df, target, protected_attr):
    groups = [
        (df[protected_attr] == 0) & (df[target] == 0),  # Female, low income
        (df[protected_attr] == 0) & (df[target] == 1),  # Female, high income
        (df[protected_attr] == 1) & (df[target] == 0),  # Male, low income
        (df[protected_attr] == 1) & (df[target] == 1)   # Male, high income
    ]
    
    min_size = min([sum(g) for g in groups])
    sampled_dfs = [df[g].sample(min_size, random_state=42) for g in groups]
    return pd.concat(sampled_dfs)

# 3. Fairness Evaluation Utilities
def evaluate_dataset_fairness(df, target, protected_attr):
    """Evaluate fairness at dataset level (before/after sampling)"""
    dataset = BinaryLabelDataset(
        df=df,
        label_names=[target],
        protected_attribute_names=[protected_attr]
    )
    metric = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=[{protected_attr: 0}],  # Female
        privileged_groups=[{protected_attr: 1}]      # Male
    )
    return {
        'Statistical Parity Difference': metric.statistical_parity_difference(),
        'Disparate Impact': metric.disparate_impact()
    }

def evaluate_model_fairness(y_true, y_pred, protected_attr):
    """Evaluate fairness of model predictions"""
    dataset_true = BinaryLabelDataset(
        df=pd.DataFrame({'y_true': y_true, 'protected': protected_attr}),
        label_names=['y_true'],
        protected_attribute_names=['protected']
    )
    dataset_pred = dataset_true.copy()
    dataset_pred.labels = y_pred.reshape(-1, 1)
    
    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=[{'protected': 0}],  # Female
        privileged_groups=[{'protected': 1}]      # Male
    )
    return {
        'Equal Opportunity Difference': metric.equal_opportunity_difference(),
        'Average Odds Difference': metric.average_odds_difference(),
        'Disparate Impact (Predictions)': BinaryLabelDatasetMetric(
            dataset_pred,
            unprivileged_groups=[{'protected': 0}],
            privileged_groups=[{'protected': 1}]
        ).disparate_impact()
    }

# 4. Main Pipeline
def main():
    # Load data
    df = load_data()
    
    # =================================================================
    # BEFORE SAMPLING (Original Data)
    # =================================================================
    print("\n" + "="*40)
    print("BEFORE UNIFORM SAMPLING (ORIGINAL DATA)")
    print("="*40)
    
    # Evaluate original dataset fairness
    orig_fairness = evaluate_dataset_fairness(df, 'income', 'sex')
    print("\nDataset Fairness Metrics (Original):")
    for metric, value in orig_fairness.items():
        print(f"{metric}: {value:.4f}")
    
    # Train model on original data
    X_orig = df.drop(columns=['income', 'fnlwgt'])
    y_orig = df['income']
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42
    )
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), 
         ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country'])
    ])
    
    model_orig = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model_orig.fit(X_train_orig, y_train_orig)
    
    # Evaluate original model
    y_pred_orig = model_orig.predict(X_test_orig)
    print("\nModel Performance (Original):")
    print(classification_report(y_test_orig, y_pred_orig))
    
    orig_model_fairness = evaluate_model_fairness(y_test_orig, y_pred_orig, X_test_orig['sex'])
    print("\nModel Fairness Metrics (Original):")
    for metric, value in orig_model_fairness.items():
        print(f"{metric}: {value:.4f}")
    
    # =================================================================
    # AFTER SAMPLING
    # =================================================================
    print("\n" + "="*40)
    print("AFTER UNIFORM SAMPLING")
    print("="*40)
    
    # Apply uniform sampling
    df_sampled = uniform_sampling(df, 'income', 'sex')
    
    # Evaluate sampled dataset fairness
    sampled_fairness = evaluate_dataset_fairness(df_sampled, 'income', 'sex')
    print("\nDataset Fairness Metrics (After Sampling):")
    for metric, value in sampled_fairness.items():
        print(f"{metric}: {value:.4f}")
    
    # Train model on sampled data
    X_sampled = df_sampled.drop(columns=['income', 'fnlwgt'])
    y_sampled = df_sampled['income']
    X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(
        X_sampled, y_sampled, test_size=0.2, random_state=42
    )
    
    model_sampled = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model_sampled.fit(X_train_sampled, y_train_sampled)
    
    # Evaluate sampled model
    y_pred_sampled = model_sampled.predict(X_test_sampled)
    print("\nModel Performance (After Sampling):")
    print(classification_report(y_test_sampled, y_pred_sampled))
    
    sampled_model_fairness = evaluate_model_fairness(y_test_sampled, y_pred_sampled, X_test_sampled['sex'])
    print("\nModel Fairness Metrics (After Sampling):")
    for metric, value in sampled_model_fairness.items():
        print(f"{metric}: {value:.4f}")
    
    # =================================================================
    # IMPROVEMENT COMPARISON
    # =================================================================
    print("\n" + "="*40)
    print("IMPROVEMENT COMPARISON")
    print("="*40)
    
    print("\nDataset Fairness Improvement:")
    for metric in orig_fairness:
        improvement = sampled_fairness[metric] - orig_fairness[metric]
        print(f"{metric}: {improvement:+.4f} (Before: {orig_fairness[metric]:.4f}, After: {sampled_fairness[metric]:.4f})")
    
    print("\nModel Fairness Improvement:")
    for metric in orig_model_fairness:
        improvement = sampled_model_fairness[metric] - orig_model_fairness[metric]
        print(f"{metric}: {improvement:+.4f} (Before: {orig_model_fairness[metric]:.4f}, After: {sampled_model_fairness[metric]:.4f})")

if __name__ == "__main__":
    main()