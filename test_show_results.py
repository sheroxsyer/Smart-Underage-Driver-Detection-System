import pytest
from flask import Flask
from app import create_app, db  # Import your app factory and database initialization

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()  # Create all tables in the in-memory database
        yield client
        with app.app_context():
            db.drop_all()  # Drop tables after tests

def test_show_results(client):
    # Arrange: Create test data in the database
    from app.models import Result, Location  # Import your models
    location1 = Location(location_name="Test Location")
    db.session.add(location1)
    db.session.commit()
    
    result1 = Result(location_name="Test Location", date_time="2024-12-10 10:00:00", image_path="static/test_image.jpg")
    db.session.add(result1)
    db.session.commit()
    
    # Act: Send a GET request to the show_results route
    response = client.get('/show_results')

    # Assert: Check that the response is successful and contains the expected content
    assert response.status_code == 200
    assert b'Results' in response.data
    assert b'Test Location' in response.data
    assert b'test_image.jpg' in response.data
