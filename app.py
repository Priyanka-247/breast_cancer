import matplotlib
matplotlib.use('Agg')
import os
import sys

# Setup module search paths dynamically
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DASH_DIR = os.path.join(BASE_DIR, 'dash_dashboard')
SRC_PYTHON_DIR = os.path.join(BASE_DIR, 'src', 'python')

for p in [BASE_DIR, DASH_DIR, SRC_PYTHON_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dash_dashboard.app import app as dash_app

# Vercel WSGI entrypoint callable
app = dash_app.server
