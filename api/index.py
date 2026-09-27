import matplotlib
matplotlib.use('Agg')
import os
import sys

# Ensure root directory, dash_dashboard, and src/python are in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dash_dir = os.path.join(root_dir, 'dash_dashboard')
src_dir = os.path.join(root_dir, 'src', 'python')

for d in [root_dir, dash_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from dash_dashboard.app import app as dash_app

class VercelDashMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path_info = environ.get('PATH_INFO', '')
        if path_info.startswith('/api/index'):
            environ['PATH_INFO'] = path_info.replace('/api/index', '', 1) or '/'
        elif path_info.startswith('/api'):
            environ['PATH_INFO'] = path_info.replace('/api', '', 1) or '/'
        return self.wsgi_app(environ, start_response)

# Expose WSGI app wrapped with Vercel path normalization middleware
app = VercelDashMiddleware(dash_app.server)
