"""
Mac Server UI - CLI and optional web interface for displaying pairing information.
"""
import asyncio
import sys
import os
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.main import app, pairing_manager
from server.pairing import PairingManager

# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class PairingWebHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for pairing web UI."""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Get current pairing info
            pairing_code, session_token, qr_code = pairing_manager.generate_pairing_code()
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Forge Engine - Device Pairing</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        min-height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }}
                    .container {{
                        background: rgba(255, 255, 255, 0.1);
                        backdrop-filter: blur(10px);
                        border-radius: 20px;
                        padding: 40px;
                        text-align: center;
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                    }}
                    h1 {{
                        margin-top: 0;
                        font-size: 2.5em;
                    }}
                    .pairing-code {{
                        font-size: 4em;
                        font-weight: bold;
                        letter-spacing: 0.2em;
                        margin: 20px 0;
                        font-family: 'Courier New', monospace;
                        background: rgba(255, 255, 255, 0.2);
                        padding: 20px;
                        border-radius: 10px;
                    }}
                    .qr-code {{
                        margin: 20px 0;
                    }}
                    .qr-code img {{
                        max-width: 300px;
                        border-radius: 10px;
                        background: white;
                        padding: 20px;
                    }}
                    .info {{
                        margin-top: 20px;
                        font-size: 0.9em;
                        opacity: 0.8;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🔗 Device Pairing</h1>
                    <p>Scan the QR code or enter the code below on your mobile device</p>
                    <div class="pairing-code">{pairing_code}</div>
                    <div class="qr-code">
                        <img src="{qr_code}" alt="QR Code">
                    </div>
                    <div class="info">
                        Code expires in {PairingManager.PAIRING_CODE_EXPIRY_MINUTES} minutes
                    </div>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress log messages


def print_pairing_info(pairing_code: str, qr_code: str):
    """Print pairing information to terminal."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}🔗 Forge Engine - Device Pairing{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
    print(f"{Colors.OKGREEN}Pairing Code:{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{pairing_code}{Colors.ENDC}\n")
    print(f"{Colors.WARNING}Scan the QR code with your mobile device{Colors.ENDC}")
    print(f"{Colors.WARNING}Or enter the code manually{Colors.ENDC}\n")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def start_web_server(port: int = 8080):
    """Start a simple web server for pairing UI."""
    server = HTTPServer(('localhost', port), PairingWebHandler)
    print(f"{Colors.OKGREEN}Web UI available at: http://localhost:{port}{Colors.ENDC}")
    
    def run_server():
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return server


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Forge Engine Server UI')
    parser.add_argument('--web', action='store_true', help='Start web UI on port 8080')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--web-port', type=int, default=8080, help='Web UI port (default: 8080)')
    
    args = parser.parse_args()
    
    # Generate initial pairing code
    pairing_code, session_token, qr_code = pairing_manager.generate_pairing_code()
    
    # Print to terminal
    print_pairing_info(pairing_code, qr_code)
    
    # Start web server if requested
    if args.web:
        server = start_web_server(args.web_port)
        webbrowser.open(f'http://localhost:{args.web_port}')
        print(f"{Colors.OKGREEN}Web browser opened automatically{Colors.ENDC}\n")
    
    # Start FastAPI server
    print(f"{Colors.OKGREEN}Starting FastAPI server on port {args.port}...{Colors.ENDC}\n")
    
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

