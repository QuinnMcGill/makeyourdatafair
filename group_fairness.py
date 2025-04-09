from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd

def eval_group_fairness(data, target, protected_attr, mode='dataset', y_pred=None, privileged_val=1, favorable_val=1):
    """
    Evaluates group-level fairness metrics for the protected attribute.

    Parameters:
        data (pd.DataFrame): Dataset including the target and protected attribute.
        target (str): The target column name (e.g., 'Income Binary').
        protected_attr (str): The protected attribute column name (e.g., 'sex').
        mode (str): Evaluation mode - 'dataset' to assess fairness of the data, 
                    'model' to assess fairness of model predictions. Default is 'dataset'.
        y_pred (array-like, optional): Required if mode='model'. The model's predictions.
        privileged_val (int): Value that represents the privileged group (default = 1).
        favorable_val (int): Value that represents a favorable outcome (default = 1).

    Returns:
        dict: Dictionary containing fairness metric results:
              - Statistical Parity Difference
              - Disparate Impact
              - Demographic Parity
    """
    if mode not in ['dataset', 'model']:
        raise ValueError("mode must be either 'dataset' or 'model'")

    df_copy = data.copy()

    if mode == 'model':
        if y_pred is None:
            raise ValueError("y_pred must be provided when mode='model'")
        df_copy[target] = y_pred  # Replace label column with model predictions
        dataset = BinaryLabelDataset(
            favorable_label=favorable_val,
            unfavorable_label=1 - favorable_val,
            df=df_copy,
            label_names=[target],
            protected_attribute_names=[protected_attr]
        )
    else:  # mode == 'dataset'
        dataset = StandardDataset(
            df_copy,
            label_name=target,
            protected_attribute_names=[protected_attr],
            favorable_classes=[favorable_val],
            privileged_classes=[[privileged_val]]
        )

    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=[{protected_attr: privileged_val}],
        unprivileged_groups=[{protected_attr: 1 - privileged_val}]
    )

    fairness_results = {
        "Statistical Parity Difference": metric.statistical_parity_difference(),
        "Disparate Impact": metric.disparate_impact(),
        "Demographic Parity": metric.mean_difference()
    }

    return fairness_results