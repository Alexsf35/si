import numpy as np

def rmse(y_true:list, Y_pred:list) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between true values and predictions.

    Parameters
    ----------
    y_true : list
        List of true target values.
    y_pred : list
        List of predicted values.

    Returns
    -------
    float
        The RMSE value, rounded to 3 decimal places.
    """
    y_true=np.array(y_true)
    Y_pred=np.array(Y_pred)

    return float(round(np.sqrt(sum(np.square(y_true-Y_pred))/len(y_true)),3))
