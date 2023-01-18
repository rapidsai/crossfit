import panel as pn
from crossfit.dashboard.template import TremorDashboard
from crossfit.dashboard.components import *

pn.extension(sizing_mode="stretch_both")


auc_metric = Card(Text("AUC"), Metric("0.9"), decoration="top")

dashboard = TremorDashboard(site_title="Model Performance")
dashboard.main.append(auc_metric)
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
