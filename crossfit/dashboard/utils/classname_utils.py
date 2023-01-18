from typing import List

from crossfit.dashboard.utils.color_variant_mapping import (
    ColorTypes,
    colorVariantMapping,
)
from crossfit.dashboard.utils.colors import twColorsHex


def classNames(classes: List[str]) -> str:
    return " ".join(filter(None, classes))


def getPixelsFromTwClassName(twClassName) -> int:
    classNameParts = twClassName.split("-")
    return int(classNameParts[-1]) * 4


def getColorVariantsFromColorThemeValue(colorThemeValue: str) -> ColorTypes:
    colorThemeValueParts = colorThemeValue.split("-")
    baseColor = colorThemeValueParts[0]
    colorValue = colorThemeValueParts[1]
    colorVariants = colorVariantMapping[baseColor][colorValue]
    return colorVariants


def getHexFromColorThemeValue(colorThemeValue: str) -> str:
    colorThemeValueParts = colorThemeValue.split("-")
    if not colorThemeValue or len(colorThemeValueParts) != 2:
        return ""
    baseColor = colorThemeValueParts[0]
    hexValue = twColorsHex[baseColor][500]
    return hexValue


def parseTruncateOption(option: bool) -> str:
    return "tr-truncate" if option else ""


def parseHFullOption(option: bool) -> str:
    return "tr-h-full" if option else ""


def parseWFullOption(option: bool) -> str:
    return "tr-w-full" if option else ""
