import numpy as np
from sklearn.metrics import confusion_matrix

def eval_equality(x_test_scaled, y_pred, sensitive_attribute_index, y_test):
    metrics = {'TPR': {}, 'FPR': {}}
    
    groups = np.unique(x_test_scaled[:, sensitive_attribute_index])  
    
    for group in groups:
        group_mask = x_test_scaled[:, sensitive_attribute_index] == group
        group_y_pred = y_pred[group_mask]
        group_y_test = y_test[group_mask]
        
        # Confusion Matrix 
        tn, fp, fn, tp = confusion_matrix(group_y_test, group_y_pred).ravel()
        
        # True Positive Rate (TPR) = TP / (TP + FN)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False Positive Rate (FPR) = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Store metrics for the group
        metrics['TPR'][group] = tpr
        metrics['FPR'][group] = fpr
    
    # Calculate Equality of Opportunity (EO) - based on TPR
    tpr_diff = abs(metrics['TPR'][groups[0]] - metrics['TPR'][groups[1]])
    
    # Calculate Equality of Odds (EOd) - based on both TPR and FPR
    fpr_diff = abs(metrics['FPR'][groups[0]] - metrics['FPR'][groups[1]])
    eod = tpr_diff + fpr_diff
    
    return tpr_diff, eod
