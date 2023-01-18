import panel as pn
from crossfit.dashboard.template import TremorDashboard
from crossfit.dashboard.components import *

pn.extension(sizing_mode="stretch_both", raw_css=[])


auc_metric = Card(Text("AUC"), Metric("0.9"), decoration="top")


def test_basic():
    dashboard = TremorDashboard(site_title="Model Performance")
    dashboard.main.append(auc_metric)

    a = 5
