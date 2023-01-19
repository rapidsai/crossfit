import pandera as pa
from pandera.typing import Series

from crossfit.dashboard.components.layout import ColGrid, Card
from crossfit.dashboard.components.metrics import MetricOverviewCard


class ComparisonSchema(pa.SchemaModel):
    name: Series[str] = pa.Field()
    color: Series[str] = pa.Field()
    top_metric_name: Series[str] = pa.Field()
    top_metric_value: Series[float] = pa.Field()
    metrics: Series[object] = pa.Field()
    
    
    
def MetricOverviewCompareCard(
    current,
    reference,
    threshold=0.5, 
    progress_main_color="blue",
    **kwargs
) -> Card:
    ...


# @pa.check_input(ComparisonSchema)
def TopMetricCompare(
    df: ComparisonSchema
) -> ColGrid:
    cols = []
    
    for _, row in df.iterrows():
        cols.append(MetricOverviewCard(
            row["name"],
            row["top_metric_name"],
            row["top_metric_value"],
            row["metrics"],            
            decoration="top",
            decoration_color=row["color"]
        ))
    
    return ColGrid(
        *cols,
        num_cols_lg=len(df),
        margin_top="mt-4",
        gap_x="gap-x-4",
        gap_y="gap-y-4"
    )