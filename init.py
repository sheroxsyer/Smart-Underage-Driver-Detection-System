from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config_test import Config

db = SQLAlchemy()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    # Import routes, models, etc.
    from app import routes, models

    return app
