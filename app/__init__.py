"""TerraTech - Time Series Forecasting Web Application."""
from flask import Flask
from app.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, TEMPLATES_DIR, STATIC_DIR
from app.routes import create_routes


def create_app():
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=TEMPLATES_DIR,
        static_folder=STATIC_DIR
    )
    create_routes(app)
    return app


# Default app instance for WSGI servers
app = create_app()


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)