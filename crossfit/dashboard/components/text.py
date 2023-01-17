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
    styles = " ".format([
        "text-elem tremor-base",
        lib.parseTruncateOption(truncate),
        "tr-whitespace-nowrap" if truncate else "tr-shrink-0",
        lib.parseHeight(tr_height) if tr_height else "",
        "tr-overflow-y-auto" if tr_height else "",
        lib.parseMarginTop.get(margin_top),
        lib.parseTextAlignment.get(text_alignment),
        lib.getColorVariantsFromColorThemeValue(lib.getColorTheme(color)["text"]).textColor,
        lib.FontSize.sm,
        lib.FontWeight.sm
    ])
    
    print("AAAA")
    
    return pn.pane.HTML(f"<p class=\"{styles}\">{text}</p>", **kwargs)



# class Text(pn.pane.HTML):
#     ...
    
    # def __init__(
    #     self, 
    #     color=lib.BaseColor.Gray, 
    #     text_alignment=lib.TextAlignment.Left, 
    #     truncate=False, 
    #     tr_height="", 
    #     margin_top="mt-0", 
    #     **params
    # ):
    #     self.color = color
    #     self.text_alignment = text_alignment
    #     self.truncate = truncate
    #     self.tr_height = tr_height
    #     self.margin_top = margin_top
    #     self.params = params
    #     super().__init__(**params)
        
    # def _css_classes(self):
    #     return super()._css_classes() + [
    #         "text-elem tremor-base",
    #         lib.parseTruncateOption(self.truncate),
    #         "tr-whitespace-nowrap" if self.truncate else "tr-shrink-0",
    #         lib.parseHeight(self.tr_height) if self.tr_height else "",
    #         "tr-overflow-y-auto" if self.tr_height else "",
    #         lib.parseMarginTop(self.margin_top),
    #         lib.parseTextAlignment(self.text_alignment),
    #         lib.getColorVariantsFromColorThemeValue(lib.getColorTheme(self.color)).textColor,
    #         lib.fontSize["sm"],
    #         lib.fontWeight["sm"]
    #     ]
