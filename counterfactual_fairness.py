from sklearn.preprocessing import StandardScaler
def evaluate_counterfactual_fairness_sex(model, X):
    """
    Evaluates counterfactual fairness by flipping the 'sex' attribute.
    """
    # Store the original predictions on the test set
    original_predictions = model.predict(StandardScaler().fit_transform(X))

    # Create a copy of the dataset
    X_counterfactual = X.copy()

    # Flip 'sex' (0 -> 1, 1 -> 0)
    if 'sex' in X.columns:
        X_counterfactual['sex'] = 1 - X_counterfactual['sex']
    else:
        raise KeyError(f"The sensitive attribute 'sex' is not present in the dataset.")

    # Get counterfactual predictions
    counterfactual_predictions = model.predict(StandardScaler().fit_transform(X_counterfactual))

    # Compare the original and counterfactual predictions
    comparison = pd.DataFrame({
        'original': original_predictions,
        'counterfactual': counterfactual_predictions,
        'same_decision': original_predictions == counterfactual_predictions
    })

    # Return the comparison dataframe
    return comparison