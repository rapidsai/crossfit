import pandera as pa
from pandera.typing import Series

from crossfit.dashboard.components.layout import Card, ColGrid, Block
from crossfit.dashboard.components.text import Text, Metric
from crossfit.dashboard.components.list import List, ListItem
from crossfit.dashboard.components.bar import ProgressBar
import crossfit.dashboard.utils as lib



def MetricCard(
    title, 
    value, 
    is_percentage=False, 
    threshold=0.5, 
    html=True, 
    progress_main_color="blue",
    **kwargs
) -> Card:
    text = Text(title)
    if is_percentage:
        if value < 1:
            value = value * 100            
        val = Metric(f"{value}%")
    else:
        val = Metric(str(value) if value <= 1 else f"{value}%")
    if value >= 1:
        threshold = threshold * 100
    progress = ProgressBar(
        value, 
        color=progress_main_color if value >= threshold else "red", 
        margin_top="mt-2"
    )
    
    return Card(text, val, progress, html=html, **kwargs)


def MetricOverviewCard(
    name,
    top_metric_name,
    top_metric_value,
    metrics,
    threshold=0.5, 
    progress_main_color="blue",
    **kwargs
) -> Card:
    list_items = []
    
    for row in metrics:
        if row["value"] > 1:
            _threshold = threshold * 100
        list_items.append(ListItem(
            Block(
                Text(row["name"]),
                ProgressBar(
                    row["value"], 
                    label=f"{row['value']}%" if row["value"] > 1 else str(row["value"]),
                    margin_top="mt-1",
                    color=progress_main_color if row["value"] >= _threshold else "red", 
                )
            )
        ))
    
    return Card(
        Text(f"{name} • {top_metric_name}"),
        Metric(top_metric_value),
        List(*list_items, margin_top="mt-4"),
        html=True,
        **kwargs
    )


class MetricSchema(pa.SchemaModel):
    name: Series[str] = pa.Field()
    value: Series[float] = pa.Field()


@pa.check_input(MetricSchema)
def TopMetricCards(
    df: MetricSchema,
    threshold=0.5
) -> ColGrid:
    cols = []
    for _, row in df.iterrows():
        col = MetricCard(
            row["name"], 
            row["value"], 
            threshold=threshold, 
            html=True
        )
        cols.append(col)
    
    return ColGrid(
        *cols,
        num_cols_md=2,
        num_cols_lg=4,
        gap_x="gap-x-4", 
        gap_y="gap-y-4",
        margin_top="mt-4"
    )