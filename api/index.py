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
    # If request contains debug keyword anywhere, output plain text environment dump
    req_str = str(environ)
    if 'debug' in environ.get('PATH_INFO', '') or 'debug' in environ.get('QUERY_STRING', '') or 'debug' in req_str:
        lines = [f"{k} = {v}" for k, v in sorted(environ.items())]
        dump = "\n".join(lines)
        start_response('200 OK', [('Content-Type', 'text/plain')])
        return [dump.encode('utf-8')]

    path = environ.get('PATH_INFO', '')
    if path.startswith('/api/index.py'):
        path = path[13:] or '/'
    elif path.startswith('/api/index'):
        path = path[10:] or '/'
    elif path.startswith('/api'):
        path = path[4:] or '/'

    environ['PATH_INFO'] = path or '/'
    return _original_wsgi_app(environ, start_response)

app.wsgi_app = _custom_wsgi_app
