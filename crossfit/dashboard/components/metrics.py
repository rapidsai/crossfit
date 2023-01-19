import panel as pn

from crossfit.dashboard.components.layout import Card
from crossfit.dashboard.components.text import Text, Metric
import crossfit.dashboard.utils as lib
from crossfit.array.ops import percentile


def ProgressBar(
    value,
    # label=None,
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
    
    html = f"""<div class=\"{outer_classes}\">
        <div class=\"{inner_classes}\">
            <div class=\"{bar_classes}\" style=\"{bar_styles}\"></div>
        </div>
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
):
    text = Text(title)
    if is_percentage:
        if value < 1:
            value = value * 100
            threshold = threshold * 100
        val = Metric(f"{value}%")
    else:
        val = Metric(str(value))
    progress = ProgressBar(
        value, 
        color=progress_main_color if value >= threshold else "red", 
        margin_top="mt-2"
    )
    
    return Card(text, val, progress, html=html, **kwargs)


def TopMetricCards(
    df,
        
):
    ...