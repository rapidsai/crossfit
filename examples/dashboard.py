import panel as pn
from crossfit import dashboard as dash

pn.extension(sizing_mode='stretch_both', raw_css=[])


auc_metric = dash.Card(
    dash.Text("AUC"),
    dash.Metric("0.9"),
    decoration="top"
)

dashboard = dash.TremorDashboard(site_title="Model Performance")
dashboard.main.append(
    auc_metric
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