from enum import Enum


class StrEnum(Enum):
    def __str__(self) -> str:
        return self.value


class TextAlignment(StrEnum):
    Left = "text-left"
    Center = "text-center"
    Right = "text-right"
    Justify = "text-justify"
    Start = "text-start"
    End = "text-end"


class DeltaType(StrEnum):
    Increase = "increase"
    ModerateIncrease = "moderateIncrease"
    Decrease = "decrease"
    ModerateDecrease = "moderateDecrease"
    Unchanged = "unchanged"


class BaseColor(StrEnum):
    Slate = "slate"
    Gray = "gray"
    Zinc = "zinc"
    Neutral = "neutral"
    Stone = "stone"
    Red = "red"
    Orange = "orange"
    Amber = "amber"
    Yellow = "yellow"
    Lime = "lime"
    Green = "green"
    Emerald = "emerald"
    Teal = "teal"
    Cyan = "cyan"
    Sky = "sky"
    Blue = "blue"
    Indigo = "indigo"
    Violet = "violet"
    Purple = "purple"
    Fuchsia = "fuchsia"
    Pink = "pink"
    Rose = "rose"


class Size(StrEnum):
    XS = "xs"
    SM = "sm"
    MD = "md"
    LG = "lg"
    XL = "xl"


class Importance(StrEnum):
    Primary = "primary"
    Secondary = "secondary"


class ButtonVariant(StrEnum):
    Primary = "primary"
    Secondary = "secondary"
    Light = "light"


class HorizontalPosition(StrEnum):
    Left = "left"
    Right = "right"


class VerticalPosition(StrEnum):
    Top = "top"
    Bottom = "bottom"
