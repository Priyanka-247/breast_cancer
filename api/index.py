import os
import sys

# Ensure root directory, dash_dashboard, and src/python are in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dash_dir = os.path.join(root_dir, 'dash_dashboard')
src_dir = os.path.join(root_dir, 'src', 'python')

for d in [root_dir, dash_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

def app(environ, start_response):
    try:
        import matplotlib
        matplotlib.use('Agg')
        from dash_dashboard.app import app as dash_app
        return dash_app.server(environ, start_response)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        start_response('200 OK', [('Content-Type', 'text/plain')])
        return [f"Serverless Traceback Error:\n{err_msg}".encode('utf-8')]
