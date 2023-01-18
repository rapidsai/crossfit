from pathlib import Path

import panel as pn
from panel.template.base import BasicTemplate
from panel.config import config
from panel.io.resources import CSS_URLS


from crossfit.dashboard import lib
from crossfit.dashboard.components import (
    Text,
    Card,
    Metric
)



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
    
    _modifiers = {
        pn.Card: {
            'children': {'margin': (10, 10)},
            'button_css_classes': ['card-button'],
            'margin': (10, 5),
        },
    }

    
    _css = MAIN_CSS
    _template = STATIC / 'tremor.html'


__all__ = [
    "TremorDashboard",
    "lib",
    "Text",
    "Card",
    "Metric"
]