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


badge_proportions_icon_only = {
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
        "paddingTop": lib.spacing["twoXs"]["paddingTop"],
        "paddingBottom": lib.spacing["twoXs"]["paddingBottom"],
        "fontSize": lib.FontSize.sm,
    },
    "md": {
        "paddingLeft": lib.spacing["lg"]["paddingLeft"],
        "paddingRight": lib.spacing["lg"]["paddingRight"],
        "paddingTop": lib.spacing["xs"]["paddingTop"],
        "paddingBottom": lib.spacing["xs"]["paddingBottom"],
        "fontSize": lib.FontSize.md,
    },
    "lg": {
        "paddingLeft": lib.spacing["xl"]["paddingLeft"],
        "paddingRight": lib.spacing["xl"]["paddingRight"],
        "paddingTop": lib.spacing["xs"]["paddingTop"],
        "paddingBottom": lib.spacing["xs"]["paddingBottom"],
        "fontSize": lib.FontSize.lg,
    },
    "xl": {
        "paddingLeft": lib.spacing["xl"]["paddingLeft"],
        "paddingRight": lib.spacing["xl"]["paddingRight"],
        "paddingTop": lib.spacing["xs"]["paddingTop"],
        "paddingBottom": lib.spacing["xs"]["paddingBottom"],
        "fontSize": lib.FontSize.xl,
    },
}

badge_proportions_with_text = {
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


colors = {
    lib.DeltaType.Increase: {
        "bgColor": lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(lib.BaseColor.Emerald)["lightBackground"]
        ).bgColor,
        "textColor": lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(lib.BaseColor.Emerald)["darkText"]
        ).textColor,
    },
    lib.DeltaType.ModerateIncrease: {
        "bgColor": lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(lib.BaseColor.Emerald)["lightBackground"]
        ).bgColor,
        "textColor": lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(lib.BaseColor.Emerald)["darkText"]
        ).textColor,
    },
    lib.DeltaType.Decrease: {
        "bgColor": lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(lib.BaseColor.Rose)["lightBackground"]
        ).bgColor,
        "textColor": lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(lib.BaseColor.Rose)["darkText"]
        ).textColor
    },
    lib.DeltaType.ModerateDecrease: {
        "bgColor": lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(lib.BaseColor.Rose)["lightBackground"]
        ).bgColor,
        "textColor": lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(lib.BaseColor.Rose)["darkText"]
        ).textColor,
    },
    lib.DeltaType.Unchanged: {
        "bgColor": lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(lib.BaseColor.Orange)["lightBackground"]
        ).bgColor,
        "textColor": lib.getColorVariantsFromColorThemeValue(
            lib.getColorTheme(lib.BaseColor.Orange)["darkText"]
        ).textColor,
    },
}

delta_icons = {
    lib.DeltaType.Increase: """
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="{cls}">
        <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 10.5L12 3m0 0l7.5 7.5M12 3v18" />
    </svg>""",
    lib.DeltaType.ModerateIncrease: """
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="{cls}">
        <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 19.5l15-15m0 0H8.25m11.25 0v11.25" />
    </svg>""",
    lib.DeltaType.Decrease: """
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="{cls}">
        <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 13.5L12 21m0 0l-7.5-7.5M12 21V3" />
    </svg>""",
    lib.DeltaType.ModerateDecrease: """
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="{cls}">
        <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 4.5l15 15m0 0V8.25m0 11.25H8.25" />
    </svg>""",
    lib.DeltaType.Unchanged: """
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="{cls}">
        <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
    </svg>"""
}