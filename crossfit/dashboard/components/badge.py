import panel as pn

import crossfit.dashboard.utils as lib


badge_proportions = {
    "xs": {
        "paddingLeft": lib.spacing["sm"]["paddingLeft"],
        "paddingRight": lib.spacing["sm"]["paddingRight"],
        "paddingTop": lib.spacing["threeXs"]["paddingTop"],
        "paddingBottom": lib.spacing["threeXs"]["paddingBottom"],
        "fontSize": lib.FontSize.xs,
    },
    "sm": {
        "paddingLeft": lib.spacing["md"]["paddingLeft"],
        "paddingRight": lib.spacing["md"]["paddingRight"],
        "paddingTop": lib.spacing["threeXs"]["paddingTop"],
        "paddingBottom": lib.spacing["threeXs"]["paddingBottom"],
        "fontSize": lib.FontSize.sm,
    },
    "md": {
        "paddingLeft": lib.spacing["lg"]["paddingLeft"],
        "paddingRight": lib.spacing["lg"]["paddingRight"],
        "paddingTop": lib.spacing["threeXs"]["paddingTop"],
        "paddingBottom": lib.spacing["threeXs"]["paddingBottom"],
        "fontSize": lib.FontSize.md,
    },
    "lg": {
        "paddingLeft": lib.spacing["xl"]["paddingLeft"],
        "paddingRight": lib.spacing["xl"]["paddingRight"],
        "paddingTop": lib.spacing["threeXs"]["paddingTop"],
        "paddingBottom": lib.spacing["threeXs"]["paddingBottom"],
        "fontSize": lib.FontSize.lg,
    },
    "xl": {
        "paddingLeft": lib.spacing["twoXl"]["paddingLeft"],
        "paddingRight": lib.spacing["twoXl"]["paddingRight"],
        "paddingTop": lib.spacing["twoXs"]["paddingTop"],
        "paddingBottom": lib.spacing["twoXs"]["paddingBottom"],
        "fontSize": lib.FontSize.xl,
    },
}


icon_sizes = {
    "xs": {
        "height": lib.sizing["md"]["height"],
        "width": lib.sizing["md"]["width"],
    },
    "sm": {
        "height": lib.sizing["md"]["height"],
        "width": lib.sizing["md"]["width"],
    },
    "md": {
        "height": lib.sizing["md"]["height"],
        "width": lib.sizing["md"]["width"],
    },
    "lg": {
        "height": lib.sizing["lg"]["height"],
        "width": lib.sizing["lg"]["width"],
    },
    "xl": {
        "height": lib.sizing["xl"]["height"],
        "width": lib.sizing["xl"]["width"],
    },
}

def Badge(
    text,
    color = lib.BaseColor.Blue,
    icon="",
    size = lib.Size.SM,
    # tooltip=None,
    margin_top = "mt-0",
    **kwargs
) -> pn.pane.HTML:
    badgeSize = size if lib.isValidSize(size) else lib.Size.SM
    wrapper_classes = lib.classNames(["tremor-base", lib.parseMarginTop(margin_top)])
    badge_classes = lib.classNames([
        "tr-flex-shrink-0 tr-inline-flex tr-justify-center tr-items-center",
        lib.getColorVariantsFromColorThemeValue(lib.getColorTheme(color)["darkText"]).textColor,
        lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(color)["lightBackground"]
        ).bgColor,
        lib.borderRadius["full"]["all"],
        badge_proportions[str(badgeSize)]["paddingLeft"],
        badge_proportions[str(badgeSize)]["paddingRight"],
        badge_proportions[str(badgeSize)]["paddingTop"],
        badge_proportions[str(badgeSize)]["paddingBottom"],
        badge_proportions[str(badgeSize)]["fontSize"]
    ])
    
    html = f"""<div class=\"{wrapper_classes}\">
        <span class=\"{badge_classes}\">
            {icon}
            <p class\"text-elem tr-whitespace-nowrap\">{text}</p>
        </span>
    </div>"""
    
    return pn.pane.HTML(html, **kwargs)


