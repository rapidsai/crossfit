import panel as pn

from crossfit.dashboard.lib.primitives import BaseColor, HorizontalPosition, VerticalPosition
from crossfit.dashboard.lib.shape import border


def parseDecorationAlignment(decorationAlignment: str) -> str:
    if not decorationAlignment:
        return ""
    if decorationAlignment == HorizontalPosition.Left:
        return border["lg"]["left"]
    elif decorationAlignment == VerticalPosition.Top:
        return border["lg"]["top"]
    elif decorationAlignment == HorizontalPosition.Right:
        return border["lg"]["right"]
    elif decorationAlignment == VerticalPosition.Bottom:
        return border["lg"]["bottom"]
    else:
        return ""



def Card(
    *args,
    max_width: str = "max-w-none",
    h_full: bool = False,
    shadow: bool = True,
    decoration: str = "",
    decoration_color: str = BaseColor.Blue,
    margin_top: str = "mt-0",
    **kwargs
) -> pn.Card:
    """Cards are a fundamental building block for compositions, such as KPI cards, forms, or sections.
    
    Based on: https://github.com/tremorlabs/tremor/blob/main/src/components/layout-elements/Card/Card.tsx#L55

    Args:
        max_width (str, optional): 
            Set the maximum width of the component. Defaults to "max-w-none".
        h_full (bool, optional): 
            Set the component's height behavior. Defaults to False.
        shadow (bool, optional): 
            Control a card's shadow. Defaults to True.
        decoration (Optional[str], optional): 
            Add a decorative border to the card. Defaults to None.
        decoration_color (str, optional): 
            Set a color to the border decoration. Defaults to "blue".
        margin_top (str, optional): 
            Controls the top margin. Defaults to "mt-0".
    """
    
    css_classes = "tremor-base tr-relative tr-w-full tr-mx-auto tr-text-left tr-ring-1 tr-rounded-lg".split(" ")
    css_classes.append(max_width)
    if h_full:
        css_classes.append("tr-h-full")
    if shadow:
        css_classes.append("tr-shadow")
    # TODO: Fix this 
    # https://github.com/tremorlabs/tremor/blob/238f7ea533ac3954722af0371a3a5bb6b3f59074/src/components/layout-elements/Card/Card.tsx#L29
    # if decoration:
    #     css_classes.append(f"tr-border-{decoration_color}-400")
    #     css_classes.append(f"tr-ring-{decoration_color}-200")
    css_classes.extend("tr-ring-1 tr-mt-0 tr-max-w-none tr-bg-white tr-ring-gray-200".split(" "))
    
    
    css_classes.append(margin_top)
    
    return pn.Card(
        *args, 
        css_classes=css_classes, 
        collapsible=False,
        hide_header=True,
        **kwargs
    )