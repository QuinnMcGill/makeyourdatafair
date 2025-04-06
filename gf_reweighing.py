import aif360
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.algorithms.preprocessing import Reweighing

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

def manual_reweighing(df, target, protected_attr):
    """
    Reweighing implementation following the exact theoretical formulation:
    1. Calculate expected probability Pexp(S=s, Class=c) = P(S=s) * P(Class=c)
    2. Calculate observed probability Pobs(S=s, Class=c)
    3. Compute weights W(X) = Pexp / Pobs
    4. Return reweighted dataset
    """
    # Step 1: Calculate expected probabilities (assuming independence)
    total = len(df)
    p_s1 = len(df[df[protected_attr] == 1]) / total  # P(S=1)
    p_s0 = 1 - p_s1                                  # P(S=0)
    p_c1 = len(df[df[target] == 1]) / total          # P(Class=1)
    p_c0 = 1 - p_c1                                  # P(Class=0)
    
    # Expected probabilities for each combination
    expected_probs = {
        (1, 1): p_s1 * p_c1,
        (1, 0): p_s1 * p_c0,
        (0, 1): p_s0 * p_c1,
        (0, 0): p_s0 * p_c0
    }
    
    # Step 2: Calculate observed probabilities
    observed_counts = {
        (1, 1): len(df[(df[protected_attr] == 1) & (df[target] == 1)]),
        (1, 0): len(df[(df[protected_attr] == 1) & (df[target] == 0)]),
        (0, 1): len(df[(df[protected_attr] == 0) & (df[target] == 1)]),
        (0, 0): len(df[(df[protected_attr] == 0) & (df[target] == 0)])
    }
    
    observed_probs = {k: v/total for k, v in observed_counts.items()}
    
    # Step 3: Compute weights
    weights = {}
    for key in expected_probs:
        if observed_probs[key] > 0:
            weights[key] = expected_probs[key] / observed_probs[key]
        else:
            weights[key] = 0  # Handle division by zero
    
    # Step 4: Assign weights to each instance
    df_reweighed = df.copy()
    df_reweighed['weights'] = df.apply(
        lambda row: weights[(row[protected_attr], row[target])], 
        axis=1
    )
    
    return df_reweighed

# 3. Fairness Evaluation Utilities (unchanged)
def evaluate_dataset_fairness(df, target, protected_attr):
    """Evaluate fairness at dataset level"""
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

# 4. Main Pipeline (modified for reweighing)
def main():
    # Load data
    df = load_data()
    
    # =================================================================
    # BEFORE REWEIGHING (Original Data)
    # =================================================================
    print("\n" + "="*40)
    print("BEFORE REWEIGHING (ORIGINAL DATA)")
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
    # AFTER REWEIGHING
    # =================================================================
    print("\n" + "="*40)
    print("AFTER REWEIGHING")
    print("="*40)
    
    # Apply reweighing
    df_reweighed = manual_reweighing(df, 'income', 'sex')
    
    # Evaluate reweighed dataset fairness
    reweighed_fairness = evaluate_dataset_fairness(df_reweighed, 'income', 'sex')
    print("\nDataset Fairness Metrics (After Reweighing):")
    for metric, value in reweighed_fairness.items():
        print(f"{metric}: {value:.4f}")
    
    # Train model on reweighed data
    X_reweighed = df_reweighed.drop(columns=['income', 'fnlwgt', 'weights'])
    y_reweighed = df_reweighed['income']
    weights_reweighed = df_reweighed['weights']
    
    X_train_rw, X_test_rw, y_train_rw, y_test_rw, weights_train_rw, _ = train_test_split(
        X_reweighed, y_reweighed, weights_reweighed, test_size=0.2, random_state=42
    )
    
    model_rw = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model_rw.fit(
        X_train_rw, 
        y_train_rw, 
        classifier__sample_weight=df_reweighed.loc[X_train_rw.index, 'weights']
    )
    
    # Evaluate reweighed model
    y_pred_rw = model_rw.predict(X_test_rw)
    print("\nModel Performance (After Reweighing):")
    print(classification_report(y_test_rw, y_pred_rw))
    
    # Use original test set's 'sex' for fairness evaluation (same random_state)
    rw_model_fairness = evaluate_model_fairness(y_test_rw, y_pred_rw, X_test_orig['sex'])
    print("\nModel Fairness Metrics (After Reweighing):")
    for metric, value in rw_model_fairness.items():
        print(f"{metric}: {value:.4f}")
    
    # =================================================================
    # IMPROVEMENT COMPARISON
    # =================================================================
    print("\n" + "="*40)
    print("IMPROVEMENT COMPARISON")
    print("="*40)
    
    print("\nDataset Fairness Improvement:")
    for metric in orig_fairness:
        improvement = reweighed_fairness[metric] - orig_fairness[metric]
        print(f"{metric}: {improvement:+.4f} (Before: {orig_fairness[metric]:.4f}, After: {reweighed_fairness[metric]:.4f})")
    
    print("\nModel Fairness Improvement:")
    for metric in orig_model_fairness:
        improvement = rw_model_fairness[metric] - orig_model_fairness[metric]
        print(f"{metric}: {improvement:+.4f} (Before: {orig_model_fairness[metric]:.4f}, After: {rw_model_fairness[metric]:.4f})")

if __name__ == "__main__":
    main()