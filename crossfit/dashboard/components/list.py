import panel as pn

import crossfit.dashboard.utils as lib


@lib.html_component
def List(
     children,
     margin_top = "mt-0"
) -> pn.pane.HTML:
    classes = lib.classNames([
        "tremor-base list-element tr-w-full tr-overflow-hidden tr-divide-y",
        lib.getColorVariantsFromColorThemeValue(lib.defaultColors.text).textColor,
        lib.getColorVariantsFromColorThemeValue(lib.defaultColors.lightBorder).divideColor,
        lib.parseMarginTop(margin_top)
    ])
    
    return pn.pane.HTML(f"<ul class=\"{classes}\">{children}</ul>")


@lib.html_component
def ListItem(
    children,
    space_x=None
) -> pn.pane.HTML:
    classes = lib.classNames([
        "tr-w-full tr-flex tr-justify-between tr-items-center tr-truncate tr-tabular-nums",
        lib.parseSpaceX(space_x) if space_x else space_x,
        lib.spacing["sm"]["paddingTop"],
        lib.spacing["sm"]["paddingBottom"],
        lib.FontSize.sm
    ])
    
    return pn.pane.HTML(f"<li class=\"{classes}\">{children}</li>")
