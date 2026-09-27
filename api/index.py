import os
import sys

import matplotlib
matplotlib.use('Agg')

# Ensure root directory, dash_dashboard, and src/python are in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dash_dir = os.path.join(root_dir, 'dash_dashboard')
src_dir = os.path.join(root_dir, 'src', 'python')

for d in [root_dir, dash_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from dash_dashboard.app import app as dash_app

app = dash_app.server

# Preserve Flask app interface for Vercel's vc_init.py bootstrapper while handling path rewrites
_original_wsgi_app = app.wsgi_app

def _custom_wsgi_app(environ, start_response):
    path = environ.get('PATH_INFO', '')
    if path.startswith('/api/index'):
        environ['PATH_INFO'] = path[10:] or '/'
    elif path.startswith('/api'):
        environ['PATH_INFO'] = path[4:] or '/'
    return _original_wsgi_app(environ, start_response)

app.wsgi_app = _custom_wsgi_app
