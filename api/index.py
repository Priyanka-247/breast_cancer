import os
import sys
import json

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
    if environ.get('PATH_INFO', '').endswith('/debug-env'):
        env_dict = {k: str(v) for k, v in environ.items()}
        start_response('200 OK', [('Content-Type', 'application/json')])
        return [json.dumps(env_dict, indent=2).encode('utf-8')]

    # Retrieve original path requested by client
    # Vercel passes x-matched-path or x-invoke-path or PATH_INFO
    path = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_INVOKE_PATH') or environ.get('PATH_INFO', '')
    
    # If vercel rewritten path starts with /api/index.py or /api/index:
    if path.startswith('/api/index.py'):
        path = path[13:] or '/'
    elif path.startswith('/api/index'):
        path = path[10:] or '/'
    elif path.startswith('/api'):
        path = path[4:] or '/'

    environ['PATH_INFO'] = path or '/'
    return _original_wsgi_app(environ, start_response)

app.wsgi_app = _custom_wsgi_app
