import panel as pn

import crossfit.dashboard.utils as lib


def _label(name):
    label_p_classes = lib.classNames([
        "text-elem tr-shrink-0 tr-whitespace-nowrap tr-truncate",
        lib.FontSize.sm,
        lib.FontWeight.sm
    ])
    label_p = f"""<p class=\"{label_p_classes}\">{name}</p>"""
    label_div_classes = lib.classNames([
        "tr-w-16 tr-truncate tr-text-right",
        lib.getColorVariantsFromColorThemeValue(lib.defaultColors.darkText).textColor,
        lib.spacing["sm"]["marginLeft"]
    ])
    return f"""<div class=\"{label_div_classes}\">{label_p}</div>"""


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
    
    label_div = _label(label) if label else ""
    
    html = f"""<div class=\"{outer_classes}\">
        <div class=\"{inner_classes}\">
            <div class=\"{bar_classes}\" style=\"{bar_styles}\"></div>
        </div>
        {label_div}
    </div>"""
    
    return pn.pane.HTML(html, **kwargs)


def DeltaBar(
    percentage_value,
    is_increase_positive = True,
    label=None,
    # tooltip=None,
    show_animation = True,
    margin_top = "mt-0",
    **kwargs
) -> pn.pane.HTML:
    colors = {
        lib.DeltaType.Increase: {
            "bgColor": lib.getColorVariantsFromColorThemeValue(
                lib.colorTheme[lib.BaseColor.Emerald]["background"]
            ).bgColor,
        },
        lib.DeltaType.ModerateIncrease: {
            "bgColor": lib.getColorVariantsFromColorThemeValue(
                lib.colorTheme[lib.BaseColor.Emerald]["background"]
            ).bgColor,
        },
        lib.DeltaType.Decrease: {
            "bgColor": lib.getColorVariantsFromColorThemeValue(
                lib.colorTheme[lib.BaseColor.Rose]["background"]
            ).bgColor,
        },
        lib.DeltaType.ModerateDecrease: {
            "bgColor": lib.getColorVariantsFromColorThemeValue(
                lib.colorTheme[lib.BaseColor.Rose]["background"]
            ).bgColor,
        },
        lib.DeltaType.Unchanged: {
            "bgColor": lib.getColorVariantsFromColorThemeValue(
                lib.colorTheme[lib.BaseColor.Orange]["background"]
            ).bgColor,
        },
    }
    
    
    delta_type = lib.mapInputsToDeltaType(
        lib.DeltaType.Increase if percentage_value >= 0 else lib.DeltaType.Decrease,
        is_increase_positive
    )
    
    
    wrapper_classes = lib.classNames(["tremor-base", lib.parseMarginTop(margin_top)])
    outer_classes = lib.classNames([
        "tr-relative tr-flex tr-items-center tr-w-full",
        lib.getColorVariantsFromColorThemeValue(lib.defaultColors.background).bgColor,
        lib.sizing["xs"]["height"],
        lib.borderRadius["lg"]["all"]
    ])
    
    neg_bar, pos_bar = "", ""
    bar_style = f"width: {abs(percentage_value)}%; transition: {'all 6s' if show_animation else ''};"
    if percentage_value < 0:
        neg_bar_classes = lib.classNames([
            colors[delta_type]["bgColor"],
            lib.borderRadius["full"]["left"]
        ])
        neg_bar = f"<div class=\"{neg_bar_classes}\" style=\"{bar_style}\"></div>"
        
    neg_delta = f"<div class=\"tr-flex tr-justify-end tr-h-full tr-w-1/2\">{neg_bar}</div>"
    pos_wrapper_classes = lib.classNames([
        "tr-ring-2 tr-z-10",
        lib.getColorVariantsFromColorThemeValue(lib.defaultColors.darkBackground).bgColor,
        lib.getColorVariantsFromColorThemeValue(lib.defaultColors.white).ringColor,
        lib.sizing["md"]["height"],
        lib.sizing["twoXs"]["width"],
        lib.borderRadius["lg"]["all"]
    ])
    
    if percentage_value >= 0:
        pos_bar_classes = lib.classNames([
            colors[delta_type]["bgColor"],
            lib.borderRadius["full"]["right"]
        ])
        pos_bar = f"<div class=\"{pos_bar_classes}\" style=\"{bar_style}\"></div>"
    
    pos_delta = f"""<div class=\"{pos_wrapper_classes}\">
        <div className=\"tr-flex tr-justify-start tr-h-full tr-w-1/2\">
            {pos_bar}
        </div>
    </div>"""
    
    label_div = _label(label) if label else ""
    
    html = f"""<div class=\"{wrapper_classes}\">
        <div class=\"{outer_classes}\">
            <div class=\"{outer_classes}\">
                {neg_delta}
                {pos_delta}
            </div>
            {label_div}
        </div>
        
    </div>"""
    
    return pn.pane.HTML(html, **kwargs)