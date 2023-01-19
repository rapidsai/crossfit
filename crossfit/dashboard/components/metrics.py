import panel as pn
import pandera as pa
from pandera.typing import Series

from crossfit.dashboard.components.layout import Card, ColGrid, Block
from crossfit.dashboard.components.text import Text, Metric
from crossfit.dashboard.components.list import List, ListItem
import crossfit.dashboard.utils as lib


def ProgressBar(
    value,
    label=None,
    # tooltip=None,
    show_animation=True,
    color = lib.BaseColor.Blue,
    margin_top = "mt-0",
    **kwargs
):
    if value < 1:
        value = value * 100
    
    primaryBgColor = lib.getColorVariantsFromColorThemeValue(
        lib.getColorTheme(color)["background"]
    ).bgColor
    secondaryBgColor = lib.getColorVariantsFromColorThemeValue(
        lib.getColorTheme(color)["lightBackground"]
    ).bgColor
    outer_classes = lib.classNames([
        "tremor-base tr-flex tr-items-center tr-w-full",
        lib.parseMarginTop(margin_top)
    ])
    inner_classes = lib.classNames([
         "tr-relative tr-flex tr-items-center tr-w-full",
          secondaryBgColor,
          lib.sizing["xs"]["height"],
          lib.borderRadius["lg"]["all"]
    ])
    bar_classes = lib.classNames([
        primaryBgColor,
        "tr-flex-col tr-h-full",
        lib.borderRadius["lg"]["all"]
    ])
    bar_styles = f"width: {value}%; transition: {'all 6s' if show_animation else ''};"
    
    label_div = ""
    if label:
        label_p_classes = lib.classNames([
            "text-elem tr-shrink-0 tr-whitespace-nowrap tr-truncate",
            lib.FontSize.sm,
            lib.FontWeight.sm
        ])
        label_p = f"""<p class=\"{label_p_classes}\">{label}</p>"""
        label_div_classes = lib.classNames([
            "tr-w-16 tr-truncate tr-text-right",
            lib.getColorVariantsFromColorThemeValue(lib.defaultColors.darkText).textColor,
            lib.spacing["sm"]["marginLeft"]
        ])
        label_div = f"""<div class=\"{label_div_classes}\">{label_p}</div>"""
    
    
    html = f"""<div class=\"{outer_classes}\">
        <div class=\"{inner_classes}\">
            <div class=\"{bar_classes}\" style=\"{bar_styles}\"></div>
        </div>
        {label_div}
    </div>"""
    
    # TODO: Add label
    
    return pn.pane.HTML(html, **kwargs)
    


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
        gap_y="gap-y-4"
    )