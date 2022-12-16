import enum

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import metrics_utils as tf_mutils
from sklearn.metrics import roc_curve, auc


from crossfit.core.calculate import calculate
from crossfit.ml.classification import BinaryMetrics
from crossfit.ml.metrics import tf_metrics as tfm


y_true = np.random.randint(2, size=1000)
y_pred = np.random.rand(1000)


def test_tf_accuracy():
    state = tfm.TFAccuracy()(y_pred > 0.5, y_true)
    np.testing.assert_almost_equal(
        state.result, BinaryMetrics().prepare(y_pred, y_true).accuracy
    )
    state_df = calculate(tfm.TFAccuracy(), y_true, y_pred)
    assert isinstance(state_df.result(), pd.DataFrame)
    
    
    
def test_tf_update_confusion_matrix_variables():
    num_thresholds = 200
    even_thresholds = [
        (i + 1) * 1.0 / (num_thresholds - 1)
        for i in range(num_thresholds - 2)
    ]
    
    def calculate(thresholds, is_even=False):
        def create_var():
            return tf.Variable(tf.zeros((len(thresholds),), dtype=tf.float32))
        
        
        vars = {
            tf_mutils.ConfusionMatrix.TRUE_POSITIVES: create_var(),
            tf_mutils.ConfusionMatrix.TRUE_NEGATIVES: create_var(),
            tf_mutils.ConfusionMatrix.FALSE_POSITIVES: create_var(),
            tf_mutils.ConfusionMatrix.FALSE_NEGATIVES: create_var(),
        }
        
        tf_mutils.update_confusion_matrix_variables(
            vars,
            y_true,
            y_pred,
            thresholds,
            thresholds_distributed_evenly=is_even,
        )
        
        matrix = {key.value: tf.convert_to_tensor(val).numpy() for key, val in vars.items()}
        matrix["tpr"] = matrix["tp"] / (matrix["tp"] + matrix["fn"])
        matrix["fpr"] = matrix["fp"] / (matrix["fp"] + matrix["tn"])
        
        return matrix
        
    
    fpr, tpr, tresholds = roc_curve(y_true, y_pred)
    matrix = calculate(tresholds)
    even_matrix = calculate(even_thresholds, is_even=True)
    
    sk_auc = auc(fpr, tpr)
    tfsk_auc = auc(matrix["fpr"], matrix["tpr"])
    assert np.isclose(sk_auc, tfsk_auc, rtol=1.e-3)
    
    tf_auc_metric = tf.keras.metrics.AUC(
        num_thresholds=num_thresholds,
        curve="ROC",
        summation_method="interpolation",
    )
    tf_auc = tf_auc_metric(y_true, y_pred).numpy()
    tfnp_auc = auc_tf_like(
        even_matrix["fp"], even_matrix["tp"], even_matrix["fn"], even_matrix["tn"], len(even_thresholds)
    )
    assert np.isclose(tf_auc, tfnp_auc, rtol=1.e-3)
    
    
class AUCCurve(enum.Enum):
  ROC = 'ROC'
  PR = 'PR'


class AUCSummationMethod(enum.Enum):
  INTERPOLATION = 'interpolation'
  MAJORING = 'majoring'
  MINORING = 'minoring'


def auc_tf_like(
    fp, 
    tp, 
    fn, 
    tn,
    num_thresholds, 
    curve="ROC",
    summation_method="interpolation"
):
    if (str(curve) == "PR" and
        str(summation_method) == "INTERPOLATION"):
        return auc_pr_interpolate(fp, tp, fn, num_thresholds)
    
    recall = tp / (tp + fn)
    if str(curve) == "ROC":
      fp_rate = fp / (fp + tn)
      x = fp_rate
      y = recall
    elif str(curve) == "PR":
      precision = tp / (tp + fp)
      x = recall
      y = precision

    # Find the rectangle heights based on `summation_method`.
    if str(summation_method) == "interpolation":
      heights = (y[:num_thresholds - 1] + y[1:]) / 2.
    elif str(summation_method) == "minoring":
      heights = np.minimum(y[:num_thresholds - 1], y[1:])
    elif str(summation_method) == "majoring":
      heights = np.maximum(y[:num_thresholds - 1], y[1:])

    # Sum up the areas of all the rectangles.
    return np.nansum((x[:num_thresholds - 1] - x[1:]) * heights)

    
def auc_pr_interpolate(fp, tp, fn, num_thresholds):
    # Code adapted from: https://github.com/tensorflow/model-analysis/blob/2ecb6874b1d05f02f2823ee8cdbcda8d59894e90/tensorflow_model_analysis/metrics/confusion_matrix_metrics.py#L505-L518
    
    dtp = tp[:num_thresholds - 1] - tp[1:]
    p = tp + fp
    dp = p[:num_thresholds - 1] - p[1:]
    prec_slope = dtp / np.maximum(dp, 0)
    intercept = tp[1:] - prec_slope * p[1:]
    safe_p_ratio = np.where(
        np.logical_and(p[:num_thresholds - 1] > 0, p[1:] > 0),
        p[:num_thresholds - 1] / np.maximum(p[1:], 0), np.ones_like(p[1:]))
    pr_auc_increment = (
        prec_slope * (dtp + intercept * np.log(safe_p_ratio)) /
        np.maximum(tp[1:] + fn[1:], 0))
    
    return np.nansum(pr_auc_increment)
