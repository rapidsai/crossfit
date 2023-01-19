import panel as pn
import reacton
import reacton.ipywidgets as w

from crossfit.dashboard.template import TremorDashboard
from crossfit.dashboard.components import *

pn.extension(sizing_mode="stretch_both", raw_css=[])


auc_metric = Card(Text("AUC"), Metric("0.9"), decoration="top")


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


def test_reacton():
    b = ButtonClick()
    
    a = 5


def test_basic():
    dashboard = TremorDashboard(site_title="Model Performance")
    dashboard.main.append(auc_metric)

    a = 5
