import pandas as pd
import numpy as np

def eval_disparate_impact(x_test_priv_col, y_pred, unpriv_group_val=0, favourable_outcome=1):
  df = pd.DataFrame({
      'privileged_attribute': x_test_priv_col,
      'prediction': y_pred
  })

  # Calculate selection rates for protected and unprotected groups
  unpriveleged_group = df[df['privileged_attribute'] == unpriv_group_val]
  priveleged_group = df[df['privileged_attribute'] != unpriv_group_val]
  
  unpriveleged_selection_rate = np.mean(unpriveleged_group['prediction'] == favourable_outcome)
  priveleged_selection_rate = np.mean(priveleged_group['prediction'] == favourable_outcome)
  
  # Calculate disparate impact
  if priveleged_selection_rate == 0:
      disparate_impact = float('inf') if unpriveleged_selection_rate > 0 else 1.0
  else:
      disparate_impact = unpriveleged_selection_rate / priveleged_selection_rate

  return disparate_impact
