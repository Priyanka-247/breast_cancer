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

# Expose WSGI app for Vercel serverless runtime
app = dash_app.server
