from http.server import BaseHTTPRequestHandler
import sys
import os
import traceback

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dash_dir = os.path.join(root_dir, 'dash_dashboard')
src_dir = os.path.join(root_dir, 'src', 'python')

for d in [root_dir, dash_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

class handler(BaseHTTPRequestHandler):
    def handle_request(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from dash_dashboard.app import app as dash_app
            from werkzeug.test import Client
            from werkzeug.wrappers import Response
            
            client = Client(dash_app.server, Response)
            
            path = self.path
            if path.startswith('/api/index'):
                path = path[10:] or '/'
            elif path.startswith('/api'):
                path = path[4:] or '/'

            method = self.command.lower()

            headers = {k: v for k, v in self.headers.items()}
            body = None
            if 'Content-Length' in self.headers:
                length = int(self.headers['Content-Length'])
                body = self.rfile.read(length)

            res = getattr(client, method)(
                path,
                headers=headers,
                data=body,
                follow_redirects=True
            )

            self.send_response(res.status_code)
            for k, v in res.headers.items():
                if k.lower() not in ['content-length', 'transfer-encoding']:
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(res.get_data())

        except Exception as e:
            err_str = traceback.format_exc()
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Diagnostic Runtime Error:\n{err_str}".encode('utf-8'))

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request()

    def do_PUT(self):
        self.handle_request()

    def do_DELETE(self):
        self.handle_request()
