import panel as pn

from crossfit.dashboard import lib


def Text(
    text,
    color=lib.BaseColor.Gray, 
    text_alignment=lib.TextAlignment.Left, 
    truncate=False, 
    tr_height="", 
    margin_top="mt-0", 
    **kwargs
) -> pn.pane.HTML:
    classes = lib.classNames([
        "text-elem tremor-base",
        lib.parseTruncateOption(truncate),
        "tr-whitespace-nowrap" if truncate else "tr-shrink-0",
        lib.parseHeight(tr_height) if tr_height else "",
        "tr-overflow-y-auto" if tr_height else "",
        lib.parseMarginTop(margin_top),
        lib.parseTextAlignment(text_alignment),
        lib.getColorVariantsFromColorThemeValue(lib.getColorTheme(color)["text"]).textColor,
        lib.FontSize.sm,
        lib.FontWeight.sm
    ])
    
    return pn.pane.HTML(f"<p class=\"{classes}\">{text}</p>", **kwargs)


def Metric(
    value,
    color = lib.BaseColor.Gray,
    truncate = False,
    margin_top = "mt-0",
    **kwargs
) -> pn.pane.HTML:
    color = lib.getColorVariantsFromColorThemeValue(lib.getColorTheme(color)["darkText"])
    p_classes = lib.classNames([
        "text-elem",
        "tr-whitespace-nowrap" if truncate else "tr-shrink-0",
        lib.parseTruncateOption(truncate),
        color.textColor,
        lib.FontSize.threeXl,
        lib.FontWeight.lg
    ])
    
    html = f"""<div class=\"tremor-base {lib.parseMarginTop(margin_top)}\">
    <p class=\"{p_classes}\">
        {value}
    </p>
    </div>"""
    
    return pn.pane.HTML(html, **kwargs)
