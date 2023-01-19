import panel as pn

import crossfit.dashboard.utils as lib
from crossfit.dashboard.components import badge_styles as bs


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
        bs.badge_proportions[str(badgeSize)]["paddingLeft"],
        bs.badge_proportions[str(badgeSize)]["paddingRight"],
        bs.badge_proportions[str(badgeSize)]["paddingTop"],
        bs.badge_proportions[str(badgeSize)]["paddingBottom"],
        bs.badge_proportions[str(badgeSize)]["fontSize"]
    ])
    
    html = f"""<div class=\"{wrapper_classes}\">
        <span class=\"{badge_classes}\">
            {icon}
            <p class\"text-elem tr-whitespace-nowrap\">{text}</p>
        </span>
    </div>"""
    
    return pn.pane.HTML(html, **kwargs)


def BadgeDelta(
    text="",
    deltaType = lib.DeltaType.Increase,
    isIncreasePositive = True,
    size = lib.Size.SM,
    # tooltip=None,
    margin_top = "mt-0",
    **kwargs
) -> pn.pane.HTML:
    _deltaType = deltaType if lib.isValidDeltaType(deltaType) else lib.DeltaType.Increase
    icon = bs.delta_icons[_deltaType]
    mappedDeltaType = lib.mapInputsToDeltaType(_deltaType, isIncreasePositive)
    badgeProportions = bs.badge_proportions_with_text if text else bs.badge_proportions_icon_only
    badgeSize = size if lib.isValidSize(size) else lib.Size.SM

    wrapper_classes = lib.classNames(["tremor-base", lib.parseMarginTop(margin_top)])
    badge_classes = lib.classNames([
        "tr-flex-shrink-0 tr-inline-flex tr-justify-center tr-items-center",
        lib.borderRadius["full"]["all"],
        bs.colors[mappedDeltaType]["bgColor"],
        bs.colors[mappedDeltaType]["textColor"],
        bs.badge_proportions[str(badgeSize)]["paddingLeft"],
        bs.badge_proportions[str(badgeSize)]["paddingRight"],
        bs.badge_proportions[str(badgeSize)]["paddingTop"],
        bs.badge_proportions[str(badgeSize)]["paddingBottom"],
        bs.badge_proportions[str(badgeSize)]["fontSize"]
    ])
    icon_classes = lib.classNames([
        lib.spacing["twoXs"]["negativeMarginLeft"] if text else "",
        lib.spacing["xs"]["marginRight"] if text else "",
        bs.icon_sizes[str(badgeSize)]["height"],
        bs.icon_sizes[str(badgeSize)]["width"]
    ])
    icon = icon.format(cls=icon_classes)
    
    if text:
        text = f"<p className=\"text-elem tr-whitespace-nowrap\">{text}</p>"
    
    html = f"""<span class=\"{wrapper_classes}\">
        <span class=\"{badge_classes}\">
            {icon}
            {text}
        </span>
    </span>"""
    
    
    return pn.pane.HTML(html, **kwargs)