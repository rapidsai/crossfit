import altair as alt
import pandas as pd
from sklearn.metrics import roc_auc_score


def plot_roc_auc_curve(
    fpr, tpr, thresholds, score=None, round=2, **kwargs
) -> alt.Chart:
    """
    Plot ROC AUC curve using Altair.

    Parameters
    ----------
    fpr : array-like
        False positive rate.
    tpr : array-like
        True positive rate.
    thresholds : array-like
        Thresholds for the decision function.
    score : float
        ROC AUC score.
    cutoff : float, optional
        Cutoff probability to display. If None, no cutoff line is displayed. Default is None.
    round : int, optional
        Rounding of floats. Default is 2.

    Returns
    -------
    fig : altair.Chart
        ROC AUC curve figure.
    """

    if score is None:
        score = roc_auc_score(tpr, fpr)

    model_name = "Model"

    df = pd.DataFrame(
        {
            "False-positive-rate": fpr,
            "True-positive-rate": tpr,
            "Threshold": thresholds,
            "Model": model_name,
        }
    )

    legend = alt.Legend(
        orient="bottom", offset=10, title="Model", fillColor="white", padding=5
    )

    # Create a line chart of true positive rate (TPR) vs false positive rate (FPR)
    roc_auc = (
        alt.Chart(df)
        .encode(
            x=alt.X("False-positive-rate:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("True-positive-rate:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", legend=legend),
            tooltip=[
                alt.Tooltip("Threshold:Q", format=f".{round}f"),
                alt.Tooltip("False-positive-rate:Q", format=f".{round}f"),
                alt.Tooltip("True-positive-rate:Q", format=f".{round}f"),
            ],
        )
        .mark_line()
    )

    roc_auc.encoding.x.title = "False-positive-rate"
    roc_auc.encoding.y.title = "True-positive-rate"

    output = (roc_auc + _random_line()).properties(**kwargs)

    return output


def plot_confusion_matrix(tp, fp, fn, tn, **kwargs) -> alt.Chart:
    """Plot a confusion matrix using Altair.

    Parameters
    ----------
    tp : array-like
        True-positive count
    fp : array-like
        False-positive count
    fn : array-like
        False-negative count
    tn : array-like
        True-negative count
    **kwargs:
        Additional properties to pass to the Altair chart

    Returns:
    Altair chart displaying a heatmap of the confusion matrix,
        with absolute count and percentage values displayed as text.
    """
    total = tp + fp + fn + tn

    # Create a dataframe with the confusion matrix values
    data = {
        "y_true": ["Positive", "Positive", "Negative", "Negative"],
        "y_pred": ["Positive", "Negative", "Positive", "Negative"],
        "count": [tp, fp, fn, tn],
        "total": [total, total, total, total],
        "percentage": [tp / total, fp / total, fn / total, tn / total],
    }
    df = pd.DataFrame(data)

    base = alt.Chart(df).encode(
        alt.X("y_pred:N", scale=alt.Scale(paddingInner=0)),
        alt.Y("y_true:N", scale=alt.Scale(paddingInner=0)),
    )

    base.encoding.x.title = "Predicted"
    base.encoding.y.title = "Observed"

    # Create a heatmap chart of the confusion matrix
    cf_plot = base.mark_rect().encode(
        color=alt.Color("percentage:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=[
            alt.Tooltip("count:Q", title="Absolute Count"),
            alt.Tooltip("percentage:Q", format=".2%", title="Percentage"),
        ],
    )

    percentage = base.mark_text(fontSize=12, fontWeight="bold", dy=-10).encode(
        text=alt.Text(field="percentage", format=".2%"), color=alt.value("white")
    )

    absolute = base.mark_text(fontSize=10, dy=10).encode(
        text=alt.Text(field="count"), color=alt.value("white")
    )

    return (cf_plot + percentage + absolute).properties(**kwargs)


def _random_line(name="Random") -> alt.Chart:
    random_df = pd.DataFrame({"x": [0, 1], "y": [0, 1], "Model": name})

    return (
        alt.Chart(random_df)
        .mark_line(strokeDash=[4, 4], strokeWidth=1, color="gray")
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("Model:N", legend=alt.Legend(title="Model")),
        )
    )
