from __future__ import annotations

from pathlib import Path

try:
    from flask import Flask, jsonify, request, send_from_directory
except ImportError:  # optional dependency: pip install flask
    Flask = None

from nlp_engine import NLPEngine

HERE = Path(__file__).resolve().parent


def create_app(engine: NLPEngine | None = None):
    if Flask is None:
        raise RuntimeError("Flask is not installed: pip install flask")
    app = Flask(__name__)
    engine = engine or NLPEngine()

    @app.route('/')
    def index():
        return send_from_directory(HERE, 'frontend.html')

    @app.route('/generate_code', methods=['POST'])
    def generate_code():
        data = request.get_json(silent=True) or {}
        user_input = data.get('message', '')
        if not user_input.strip():
            return jsonify({'error': 'message is required'}), 400

        t = engine.translate(user_input, language=data.get('language'))
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
