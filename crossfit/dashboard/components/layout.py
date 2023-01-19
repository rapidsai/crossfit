from typing import Optional

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


@lib.html_component
def Block(
    children,
    max_width = "max-w-none",
    space_y = "",
    textAlignment = lib.TextAlignment.Left,
    truncate = False,
    margin_top = "mt-0",
    **kwargs
) -> pn.pane.HTML:
    classes = lib.classNames([
        "tr-w-full",
        lib.parseMaxWidth(max_width),
        lib.parseSpaceY(space_y) if space_y else space_y,
        lib.parseTextAlignment(textAlignment),
        lib.parseTruncateOption(truncate),
        "tr-whitespace-nowrap" if truncate else "",
        lib.parseMarginTop(margin_top)
    ])
    
    return pn.pane.HTML(f"<div class=\"{classes}\">{children}</div>", **kwargs)


@lib.html_component
def Flex(
  children,
  justifyContent = "justify-between",
  alignItems = "items-center",
  spaceX = "",
  truncate = False,
  marginTop = "mt-0",
  **kwargs
):
    classes = lib.classNames([
        "tr-flex tr-w-full",
        lib.parseTruncateOption(truncate),
        "tr-whitespace-nowrap" if truncate else "",
        lib.parseJustifyContent(justifyContent),
        lib.parseAlignItems(alignItems),
        lib.parseSpaceX(spaceX) if spaceX else spaceX,
        lib.parseMarginTop(marginTop)
    ])
    
    return pn.pane.HTML(f"<div class=\"{classes}\">{children}</div>", **kwargs)


@lib.html_component
def Col(
    children,
    num_col_span_sm: Optional[int] = None,
    num_col_span_md: Optional[int] = None,
    num_col_span_lg: Optional[int] = None,
    num_col_span = 1,
    **kwargs
):
    def get_col_span(num, col_span_mapping):
        return col_span_mapping.get(num, "")
        
    span_base = get_col_span(num_col_span, lib.col_span)
    span_sm = get_col_span(num_col_span_sm, lib.col_span_sm)
    span_md = get_col_span(num_col_span_md, lib.col_span_md)
    span_lg = get_col_span(num_col_span_lg, lib.col_span_lg)
        
    classes = lib.classNames([span_base, span_sm, span_md, span_lg])
    
    return pn.pane.HTML(f"<div class=\"{classes}\">{children}</div>", **kwargs)


@lib.html_component
def ColGrid(
    children,  
    num_cols_sm: Optional[int] = None,
    num_cols_md: Optional[int] = None,
    num_cols_lg: Optional[int] = None,
    num_cols = 1,
    gap_x = "gap-x-0",
    gap_y = "gap-y-0",
    margin_top = "mt-0",
    **kwargs
):
    def get_grid_cols(num_cols, grid_cols_mapping):
        if num_cols is None:
            return ""
        if num_cols not in grid_cols_mapping:
            return ""
        return grid_cols_mapping[num_cols]
        
    cols_base = get_grid_cols(num_cols, lib.grid_cols)
    cols_sm = get_grid_cols(num_cols_sm, lib.grid_cols_sm)
    cols_md = get_grid_cols(num_cols_md, lib.grid_cols_md)
    cols_lg = get_grid_cols(num_cols_lg, lib.grid_cols_lg)
    
    classes = lib.classNames([
        "tr-grid",
        cols_base,
        cols_sm,
        cols_md,
        cols_lg,
        lib.parseGapX(gap_x),
        lib.parseGapY(gap_y),
        lib.parseMarginTop(margin_top)
    ])
    
    return pn.pane.HTML(f"<div class=\"{classes}\">{children}</div>", **kwargs)


def Card(
    *args,
    max_width: str = "max-w-none",
    h_full: bool = False,
    shadow: bool = True,
    decoration: str = "",
    decoration_color: str = lib.BaseColor.Blue,
    margin_top: str = "mt-0",
    html=False,
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
    
    if html:
        if len(args) > 1:
            children = " ".join(lib.parse_html_args(*args))
        else:
            children = args
        return pn.pane.HTML(f"<div class=\"{lib.classNames(classes)}\">{children}</div>", **kwargs)

    return pn.Card(
        *args, css_classes=classes, collapsible=False, hide_header=True, **kwargs
    )
