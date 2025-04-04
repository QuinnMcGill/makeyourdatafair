from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric

def evaluate_group_fairness(data, target, protected_attr):
    """
    Evaluates group-level fairness metrics for the protected attribute.

    Parameters:
    data (pd.DataFrame): The dataset including the target and protected attribute
    target (str): The target column name
    protected_attr (str): The protected attribute column name (e.g., 'sex')

    Returns:
    dict: Dictionary with fairness metric results
    """
    dataset = StandardDataset(
        data,
        label_name=target,
        protected_attribute_names=[protected_attr],
        favorable_classes=[1],  # Income > 50K is favorable
        privileged_classes=[[1]]  # Privileged: Male (1)
    )

    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=[{protected_attr: 1}],  # Privileged: Male
        unprivileged_groups=[{protected_attr: 0}]  # Unprivileged: Female
    )

    fairness_results = {
        "Statistical Parity Difference": metric.statistical_parity_difference(),
        "Disparate Impact": metric.disparate_impact(),
        "Demographic Parity": metric.mean_difference()
    }

    return fairness_results