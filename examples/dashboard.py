import panel as pn
import numpy as np
import reacton
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from vega_datasets import data

import reacton
import reacton.ipywidgets as w



from crossfit.ml.vis.classification import plot_roc_auc_curve, plot_confusion_matrix
from crossfit.dashboard.template import TremorDashboard
from crossfit.dashboard.components import *

# Calculate the true positive rate and false positive rate at various thresholds
y_true = np.random.randint(2, size=1000)
y_pred = np.random.rand(1000)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc_score = roc_auc_score(y_true, y_pred)
tp, fp, fn, tn = confusion_matrix(y_true, y_pred > 0.5).ravel()


pn.extension('ipywidgets')
pn.extension(name="vega", sizing_mode="stretch_width")


auc_metric = Card(Text("AUC"), Metric("0.9"), decoration="top")

dashboard = TremorDashboard(title="Model Performance")
# dashboard.main.append(auc_metric)


roc = pn.pane.Vega(
    plot_roc_auc_curve(fpr, tpr, thresholds, auc_score),
    height=800, sizing_mode="stretch_width", name="AUC-ROC Curve"
)

cf = pn.pane.Vega(
    plot_confusion_matrix(tp, fp, fn, tn, width=400, height=300),
    height=800, sizing_mode="stretch_width", name="Confusion Matrix"
)


pn.widgets.Tabulator.theme = 'materialize'


@reacton.component
def ButtonClick():
    # first render, this return 0, after that, the last argument
    # of set_clicks
    clicks, set_clicks = reacton.use_state(0)
    
    def my_click_handler():
        # trigger a new render with a new value for clicks
        set_clicks(clicks+1)

    button = w.Button(description=f"Clicked {clicks} times",
                      on_click=my_click_handler)
    
    button.component
    
    return button


dashboard.main.append(Card(pn.panel(ButtonClick())))
dashboard.main.append(
    ColGrid( 
        Col(
            MetricCard("KPI 1", 0.9),
            num_col_span=1, num_col_span_lg=2
        ),
        Card(Text("Title"), Metric("KPI 2"), html=True),
        Col(
            MetricCard("KPI 3", 0.9),
        ),
        MetricCard("KPI 4", 0.4),
        Card(Text("Title"), Metric("KPI 5"), html=True),
        num_cols=1, num_cols_sm=2, num_cols_lg=3, gap_x="gap-x-2", gap_y="gap-y-2"
    )
)
dashboard.main.append(
    pn.Row(
        Card(
            Text("Confusion Matrix"),
            cf,
            height=500
        ),
        Card(
             pn.Tabs(
                ("AUC-ROC Curve", roc),
                # ("Confusion Matrix", cf),            
                ("AUC-PR Curve", roc),
                tabs_location='above',
                css_classes=["w-full"]
            ),
             height=500
        )
    )
    
    
    # Card(
    #     pn.Tabs(
    #         ("AUC-ROC Curve", roc),
    #         ("Confusion Matrix", cf),            
    #         ("AUC-PR Curve", "# css"),
    #         tabs_location='above'
    #     ),
    #     max_width="max-w-full"
    # )
)

dashboard.main.append(
    Card(
        pn.widgets.Tabulator(data.iris())
    )
)

# dashboard.sidebar.append(
#     pn.Row(
#         dash.Card(
#             dash.Text("AUC"),
#             dash.Metric("0.9"),
#             decoration="top"
#         ),
#         dash.Card("# bbb")
#     )
# )


dashboard.servable()
