import panel as pn

from crossfit.dashboard import utils as lib


def parseDecorationAlignment(alignment: str) -> str:
    if not alignment:
        return ""
    alignment = str(alignment)
    if alignment == str(lib.HorizontalPosition.Left):
        return lib.border["lg"]["left"]
    elif alignment == str(lib.VerticalPosition.Top):
        return lib.border["lg"]["top"]
    elif alignment == str(lib.HorizontalPosition.Right):
        return lib.border["lg"]["right"]
    elif alignment == str(lib.VerticalPosition.Bottom):
        return lib.border["lg"]["bottom"]
    else:
        return ""


def Card(
    *args,
    max_width: str = "max-w-none",
    h_full: bool = False,
    shadow: bool = True,
    decoration: str = "",
    decoration_color: str = lib.BaseColor.Blue,
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

    classes = (
        "tremor-base tr-relative tr-w-full tr-mx-auto tr-text-left tr-ring-1".split(" ")
    )
    extra_classes = [
        lib.parseMarginTop(margin_top),
        lib.parseHFullOption(h_full),
        lib.parseMaxWidth(max_width),
        lib.getColorVariantsFromColorThemeValue(lib.defaultColors.white).bgColor,
        lib.boxShadow["md"] if shadow else "",
        lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(decoration_color)["border"]
        ).borderColor,
        lib.getColorVariantsFromColorThemeValue(
            lib.defaultColors.lightBorder
        ).ringColor,
        parseDecorationAlignment(decoration),
        lib.spacing["threeXl"]["paddingLeft"],
        lib.spacing["threeXl"]["paddingRight"],
        lib.spacing["threeXl"]["paddingTop"],
        lib.spacing["threeXl"]["paddingBottom"],
        lib.borderRadius["lg"]["all"],
    ]
    classes.extend([c for c in extra_classes if c])

    return pn.Card(
        *args, css_classes=classes, collapsible=False, hide_header=True, **kwargs
    )
