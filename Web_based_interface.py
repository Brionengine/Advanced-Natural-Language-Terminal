from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

try:
    from flask import Flask, abort, jsonify, request, send_from_directory
except ImportError:  # optional dependency: pip install flask
    Flask = None

from nlp_engine import NLPEngine

HERE = Path(__file__).resolve().parent

# Requests must name one of these hosts. Binding to 127.0.0.1 alone does not
# stop DNS rebinding: a hostile page can point its own domain at 127.0.0.1 and
# become same-origin with this app, but its requests still carry that domain
# in the Host header.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def create_app(engine_factory: Callable[[], NLPEngine] = NLPEngine,
               allowed_hosts: Iterable[str] = LOCAL_HOSTS):
    if Flask is None:
        raise RuntimeError("Flask is not installed: pip install flask")
    app = Flask(__name__)
    allowed = {h.lower() for h in allowed_hosts}

    @app.before_request
    def check_host():
        name = request.host.lower()
        if name.startswith("["):              # IPv6 literal: [::1] or [::1]:5000
            name = name[:name.find("]") + 1]
        else:                                 # strip any :port
            name = name.split(":", 1)[0]
        if name not in allowed:
            abort(403)

    @app.route('/')
    def index():
        return send_from_directory(HERE, 'frontend.html')

    @app.route('/generate_code', methods=['POST'])
    def generate_code():
        data = request.get_json(silent=True) or {}
        user_input = data.get('message', '')
        if not user_input.strip():
            return jsonify({'error': 'message is required'}), 400

        # A fresh engine per request: one shared engine would carry every
        # client's requests in a single conversation, leaking each person's
        # input into the next person's context and output.
        try:
            t = engine_factory().translate(user_input, language=data.get('language'))
        except ValueError as exc:  # unknown language
            return jsonify({'error': str(exc)}), 400
        return jsonify({
            'code': t.code,
            'language': t.language,
            'syntax_ok': t.syntax_ok,
            'missing_packages': [d.package for d in t.missing],
            'error': t.error,
        })

    return app


if __name__ == '__main__':
    # Bound to localhost with the debugger off: Flask's debug console executes
    # arbitrary Python for anyone who can reach it.
    create_app().run(host='127.0.0.1', port=5000, debug=False)
