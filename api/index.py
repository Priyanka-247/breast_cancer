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
    # Retrieve original path requested by the client from Vercel edge headers
    raw_uri = environ.get('HTTP_X_FORWARDED_URI') or environ.get('RAW_URI') or environ.get('REQUEST_URI') or environ.get('PATH_INFO', '')
    clean_path = raw_uri.split('?')[0]
    
    if clean_path.startswith('/api/index.py'):
        clean_path = clean_path[13:] or '/'
    elif clean_path.startswith('/api/index'):
        clean_path = clean_path[10:] or '/'
    elif clean_path.startswith('/api'):
        clean_path = clean_path[4:] or '/'
        
    environ['PATH_INFO'] = clean_path or '/'
    return _original_wsgi_app(environ, start_response)

app.wsgi_app = _custom_wsgi_app
