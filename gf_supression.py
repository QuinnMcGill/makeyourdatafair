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

# 1. Load and Prepare Data (unchanged)
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

# 2. SUPPRESSION Preprocessing (NEW)
def suppression_preprocessing(df, protected_attr):
    """Remove the protected attribute column entirely."""
    return df.drop(columns=[protected_attr])

# 3. Fairness Evaluation Utilities (unchanged)
def evaluate_dataset_fairness(df, target, protected_attr):
    """Evaluate fairness at dataset level (before/after suppression)"""
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

# 4. Main Pipeline (modified for suppression)
def main():
    # Load data
    df = load_data()
    
    # =================================================================
    # BEFORE SUPPRESSION (Original Data)
    # =================================================================
    print("\n" + "="*40)
    print("BEFORE SUPPRESSION (ORIGINAL DATA)")
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
    # AFTER SUPPRESSION
    # =================================================================
    print("\n" + "="*40)
    print("AFTER SUPPRESSION (REMOVED 'sex' ATTRIBUTE)")
    print("="*40)
    
    # Apply suppression
    df_suppressed = suppression_preprocessing(df, 'sex')
    
    # Note: Can't calculate dataset fairness metrics after suppression (no 'sex' column)
    print("\nNote: Dataset fairness metrics unavailable after suppression (protected attribute removed)")
    
    # Train model on suppressed data
    X_suppressed = df_suppressed.drop(columns=['income', 'fnlwgt'])
    y_suppressed = df_suppressed['income']
    X_train_suppressed, X_test_suppressed, y_train_suppressed, y_test_suppressed = train_test_split(
        X_suppressed, y_suppressed, test_size=0.2, random_state=42
    )
    
    # Update preprocessor to exclude 'sex'
    preprocessor_suppressed = ColumnTransformer([
        ('num', StandardScaler(), ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), 
         ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country'])
    ])
    
    model_suppressed = Pipeline([
        ('preprocessor', preprocessor_suppressed),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model_suppressed.fit(X_train_suppressed, y_train_suppressed)
    
    # Evaluate suppressed model
    y_pred_suppressed = model_suppressed.predict(X_test_suppressed)
    print("\nModel Performance (After Suppression):")
    print(classification_report(y_test_suppressed, y_pred_suppressed))
    
    # Need original test set's 'sex' for fairness evaluation
    # We'll use X_test_orig['sex'] since train_test_split was done with same random_state
    suppressed_model_fairness = evaluate_model_fairness(y_test_suppressed, y_pred_suppressed, X_test_orig['sex'])
    print("\nModel Fairness Metrics (After Suppression):")
    for metric, value in suppressed_model_fairness.items():
        print(f"{metric}: {value:.4f}")
    
    # =================================================================
    # IMPROVEMENT COMPARISON
    # =================================================================
    print("\n" + "="*40)
    print("IMPROVEMENT COMPARISON")
    print("="*40)
    
    print("\nModel Fairness Improvement:")
    for metric in orig_model_fairness:
        improvement = suppressed_model_fairness[metric] - orig_model_fairness[metric]
        print(f"{metric}: {improvement:+.4f} (Before: {orig_model_fairness[metric]:.4f}, After: {suppressed_model_fairness[metric]:.4f})")

if __name__ == "__main__":
    main()