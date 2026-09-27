import os
import sys
import matplotlib
matplotlib.use('Agg')

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dash_dir = os.path.join(root_dir, 'dash_dashboard')
src_dir = os.path.join(root_dir, 'src', 'python')

for d in [root_dir, dash_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

try:
    from dash_dashboard.app import app as dash_app
    _wsgi_app = dash_app.server
    _init_error = None
except Exception as e:
    import traceback
    _init_error = traceback.format_exc()
    _wsgi_app = None

def app(environ, start_response):
    if _wsgi_app is None:
        start_response('500 Internal Server Error', [('Content-Type', 'text/plain')])
        return [f"Initialization Error:\n{_init_error}".encode('utf-8')]
    try:
        return _wsgi_app(environ, start_response)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        start_response('500 Internal Server Error', [('Content-Type', 'text/plain')])
        return [f"WSGI Runtime Traceback Error:\n{err_msg}".encode('utf-8')]
