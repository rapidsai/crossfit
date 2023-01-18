from typing import List


from crossfit.dashboard.utils.primitives import (
    BaseColor,
    Size,
    DeltaType,
    Importance,
    ButtonVariant,
)
from crossfit.dashboard.utils.colors import colorTheme


def isBaseColor(baseColor: str) -> bool:
    return baseColor in BaseColor.__members__.values()


def getColorTheme(baseColor: str, defaultColor: str = BaseColor.Blue) -> dict:
    if not baseColor or not isBaseColor(baseColor):
        return colorTheme[defaultColor]
    return colorTheme[baseColor]


def isValidSize(size: str) -> bool:
    return size in Size.__members__.values()


def isValidDeltaType(deltaType: str) -> bool:
    return deltaType in DeltaType.__members__.values()


def isValidImportance(importance: str) -> bool:
    return importance in Importance.__members__.values()


def isValidVariant(variant: str) -> bool:
    return variant in ButtonVariant.__members__.values()


def mapInputsToDeltaType(deltaType: str, isIncreasePositive: bool) -> str:
    if isIncreasePositive or deltaType == DeltaType.Unchanged:
        return deltaType
    if deltaType == DeltaType.Increase:
        return DeltaType.Decrease
    if deltaType == DeltaType.ModerateIncrease:
        return DeltaType.ModerateDecrease
    if deltaType == DeltaType.Decrease:
        return DeltaType.Increase
    if deltaType == DeltaType.ModerateDecrease:
        return DeltaType.ModerateIncrease
    return ""


def defaultValueFormatter(value: int) -> str:
    return str(value)


def sumNumericArray(arr: List[int]) -> int:
    return sum(arr)
