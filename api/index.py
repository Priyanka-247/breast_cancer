import os
import sys
from urllib.parse import parse_qs

import matplotlib
matplotlib.use('Agg')

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dash_dir = os.path.join(root_dir, 'dash_dashboard')
src_dir = os.path.join(root_dir, 'src', 'python')

for d in [root_dir, dash_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from dash_dashboard.app import app as dash_app

app = dash_app.server

_original_wsgi_app = app.wsgi_app

def _custom_wsgi_app(environ, start_response):
    query_string = environ.get('QUERY_STRING', '')
    qs = parse_qs(query_string)
    
    if 'path' in qs and qs['path']:
        route_path = qs['path'][0]
        environ['PATH_INFO'] = '/' + route_path.lstrip('/')
    
    return _original_wsgi_app(environ, start_response)

app.wsgi_app = _custom_wsgi_app
