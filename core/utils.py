import numpy as np
import pandas as pd

def rolling_mean(arr, window, *, min_periods=1, center=False):
    s = pd.Series(np.asarray(arr, dtype=float))
    return s.rolling(window=window, min_periods=min_periods, center=center).mean().to_numpy()

def rolling_vec_norm(V, window, *, norm= "l_infty", min_periods=1, center=False):
    """
    V: length-T list of numpy arrays (m,)
    Returns length-T array: || rolling_norm_t(g) ||_{norm}
    """
    df = pd.DataFrame(np.asarray(V, float))
    mu = df.rolling(window=window, min_periods=min_periods, center=center).mean()
   
    if norm == "l2":
        return np.linalg.norm(mu.to_numpy(), ord=2, axis=1)
    elif norm == "l_infty":
        return np.max(np.abs(mu.to_numpy()), axis=1)
    else:
        raise ValueError("norm must be 'l2' or 'l_infty'.")