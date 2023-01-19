import panel as pn
import numpy as np
import pandas as pd
import reacton
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from vega_datasets import data

import reacton
import reacton.ipywidgets as w



from crossfit.ml.vis.classification import plot_roc_auc_curve, plot_confusion_matrix
from crossfit.dashboard.template import TremorDashboard
from crossfit.dashboard.components import *
from crossfit.dashboard.components.bar import DeltaBar

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


df = pd.DataFrame([
    {"name": "AUC", "value": 0.9},
    {"name": "Accuracy", "value": 56.0},
    {"name": "Precision", "value": 65.0},
    {"name": "Recall", "value": 43.0},
])


overview = [    
    TopMetricCards(df),
    Card(
        Flex(
            Text("Product A"), 
            BadgeDelta("+45%"),
            Text("+$9,000 (+45%)"), 
        ),
        DeltaBar(-30), 
        html=True
    ),
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
    ),
    Card(
        pn.widgets.Tabulator(data.iris())
    )
]


comp_df = pd.DataFrame([
    {
        "name": "Model A", 
        "color": "orange", 
        "top_metric_name": "AUC", 
        "top_metric_value": 0.78,
        "metrics": [
            {
                "name": 'Accuracy',
                "value": 89,
            },
            {
                "name": 'Precision',
                "value": 64,
            },
            {
                "name": 'Recall',
                "value": 49,
            },
            {
                "name": 'F1 Score',
                "value": 72,
            },
            {
                "name": 'Log Loss',
                "value": 67,
            },
        ]
    },
    {
        "name": "Model B", 
        "color": "yellow", 
        "top_metric_name": "AUC", 
        "top_metric_value": 0.9,
        "metrics": [
            {
                "name": 'Accuracy',
                "value": 89,
            },
            {
                "name": 'Precision',
                "value": 64,
            },
            {
                "name": 'Recall',
                "value": 79,
            },
            {
                "name": 'F1 Score',
                "value": 72,
            },
            {
                "name": 'Log Loss',
                "value": 67,
            },
        ]
    },
])

comparison = [
    # Card(pn.panel(ButtonClick()), html=False),
    TopMetricCompare(comp_df)
]


main_tabs = pn.Tabs(
    ("Overview", pn.Column(*overview)),
    ("Comparison", pn.Column(*comparison))
)

dashboard.main.append(main_tabs)
# dashboard.main.append()
# dashboard.main.append(
#     ColGrid( 
#         Col(
#             MetricCard("KPI 1", 0.9),
#             num_col_span=1, num_col_span_lg=2
#         ),
#         MetricCard("KPI 3", 0.9),
        
#         MetricCard("KPI 3", 0.9),
#         MetricCard("KPI 4", 0.4),
#         MetricCard("KPI 5", 0.4),
#         num_cols_sm=3, num_cols_lg=3, gap_x="gap-x-4", gap_y="gap-y-4"
#     )
# )


# dashboard.main.append(TopMetricCards(df))
# dashboard.main.append(
#     pn.Row(
#         Card(
#             Text("Confusion Matrix"),
#             cf,
#             height=500
#         ),
#         Card(
#              pn.Tabs(
#                 ("AUC-ROC Curve", roc),
#                 # ("Confusion Matrix", cf),            
#                 ("AUC-PR Curve", roc),
#                 tabs_location='above',
#                 css_classes=["w-full"]
#             ),
#              height=500
#         )
#     )
    
    
#     # Card(
#     #     pn.Tabs(
#     #         ("AUC-ROC Curve", roc),
#     #         ("Confusion Matrix", cf),            
#     #         ("AUC-PR Curve", "# css"),
#     #         tabs_location='above'
#     #     ),
#     #     max_width="max-w-full"
#     # )
# )

# dashboard.main.append(
#     Card(
#         pn.widgets.Tabulator(data.iris())
#     )
# )

dashboard.servable()
