from pathlib import Path

import panel as pn
from panel.template.base import BasicTemplate
from panel.config import config
from panel.io.resources import CSS_URLS


from crossfit.dashboard.components import Card, Text


pn.extension(sizing_mode = 'stretch_both')


STATIC = Path(__file__).parent
# STATIC = Path("/Users/romeyn/src/notebooks/panel/static")
MAIN_CSS = str(STATIC / 'css/main.a3f02895.css')

TREMOR = "https://cdn.jsdelivr.net/npm" + "@tremor/react@1.5.0/dist/tremor-react.min.js"



class TremorDashboard(BasicTemplate):
    _resources = {
        'js': {
            'react': f"{config.npm_cdn}/react@18/umd/react.production.min.js",
            'react-dom': f"{config.npm_cdn}/react-dom@18/umd/react-dom.production.min.js",
            'babel': f"{config.npm_cdn}/babel-standalone@latest/babel.min.js",
        },
        'css': {
            'tremor-main': MAIN_CSS,
            'font-awesome': CSS_URLS['font-awesome']
        }
    }
    
    _css = MAIN_CSS
    _template = STATIC / 'tremor.html'
    
    
    
border = {
  "none": {
    "left": "tr-border-l-0",
    "top": "tr-border-t-0",
    "right": "tr-border-r-0",
    "bottom": "tr-border-b-0",
    "all": "tr-border-0",
  },
  "sm": {
    "left": "tr-border-l",
    "top": "tr-border-t",
    "right": "tr-border-r",
    "bottom": "tr-border-b",
    "all": "tr-border",
  },
  "md": {
    "left": "tr-border-l-2",
    "top": "tr-border-t-2",
    "right": "tr-border-r-2",
    "bottom": "tr-border-b-2",
    "all": "tr-border-2",
  },
  "lg": {
    "left": "tr-border-l-4",
    "top": "tr-border-t-4",
    "right": "tr-border-r-4",
    "bottom": "tr-border-b-4",
    "all": "tr-border-4",
  },
}


def _parse_decoration(decoration):
    if not decoration:
        return ""
    
    if decoration == "left":
        return border["lg"]["left"]
    elif decoration == "top":
        return border["lg"]["top"]
    elif decoration == "right":
        return border["lg"]["right"]
    elif decoration == "bottom":
        return border["lg"]["bottom"]
    
    return ""



    
    
# class Card(pn.Card):
#     collapsible = param.Boolean(default=False, doc="""
#         Whether the Card should be expandable and collapsible.""")
    
#     hide_header = param.Boolean(default=True, doc="""
#         Whether to skip rendering the header.""")
    
#     css_classes = param.List(
#         "tremor-base tr-relative tr-w-full tr-mx-auto tr-text-left tr-ring-1 tr-mt-0 tr-max-w-none tr-bg-white tr-shadow tr-border-blue-400 tr-ring-gray-200 tr-pl-6 tr-pr-6 tr-pt-6 tr-pb-6 tr-rounded-lg".split(" "), 
#         doc="""
#         CSS classes to apply to the overall Card.""")
    
    
    
dashboard = TremorDashboard(site_title="Model Performance")
dashboard.sidebar.append(
    pn.Row(
        Card(
            # pn.panel("# bbb")
            Text("Some text"), 
        ),
        Card("# bbb")
    )
)


dashboard.servable()