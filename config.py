import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'  # Use an environment variable or a default value
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class DevelopmentConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or 'mysql+mysqlconnector://sheri:sheri123@localhost/smart_underage_driver_detector'
    DEBUG = True

class TestConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or 'sqlite:///:memory:'  # Use an in-memory SQLite database for tests
    TESTING = True
    DEBUG = True  # Optional: set to True for more detailed error logs during testing

class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('PROD_DATABASE_URL') or 'mysql+mysqlconnector://sheri:sheri123@prod_host/smart_underage_driver_detector'
    DEBUG = False
