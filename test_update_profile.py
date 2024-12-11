import pytest
from app import create_app, db
from models import User  # Ensure the User model is correctly imported

# The client fixture that sets up the Flask app, in-memory database, and client
@pytest.fixture
def client():
    # Create the Flask app
    app = create_app()

    # Configure the app for testing
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'  # Use in-memory database for tests
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Setup database within app context
    with app.app_context():
        db.create_all()  # Create all tables needed for testing

    # Provide the client for testing
    with app.test_client() as client:
        yield client  # Yield the client for tests

    # Cleanup: Drop all tables after tests
    with app.app_context():
        db.drop_all()


# Example test case for updating profile
def test_update_profile_success(client):
    """
    Test case for successful profile update.
    """
    # Arrange: Create a test user in the database
    with client.application.app_context():
        test_user = User(name="John Doe", email="john.doe@example.com", phone="1234567890", password="password123")
        db.session.add(test_user)
        db.session.commit()

    # Act: Send a POST request to update the user's profile
    response = client.post(
        '/update_profile',
        data={
            'name': 'Jane Doe',
            'email': 'jane.doe@example.com',
            'phone': '0987654321',
            'password': ''  # No password update
        },
        follow_redirects=True
    )

    # Assert: Check the response and database update
    assert response.status_code == 200
    assert b'Profile updated successfully' in response.data

    # Check if the database has been updated
    with client.application.app_context():
        updated_user = User.query.filter_by(email='jane.doe@example.com').first()
        assert updated_user is not None
        assert updated_user.name == 'Jane Doe'
        assert updated_user.phone == '0987654321'
        assert updated_user.password == 'password123'  # Password should remain unchanged


# Example test case for invalid email during profile update
def test_update_profile_invalid_email(client):
    """
    Test case for invalid email during profile update.
    """
    # Arrange: Create a test user in the database
    with client.application.app_context():
        test_user = User(name="John Doe", email="john.doe@example.com", phone="1234567890", password="password123")
        db.session.add(test_user)
        db.session.commit()

    # Act: Send a POST request with an invalid email
    response = client.post(
        '/update_profile',
        data={
            'name': 'Jane Doe',
            'email': 'invalid-email',
            'phone': '0987654321',
            'password': ''
        },
        follow_redirects=True
    )

    # Assert: Check the response and database state
    assert response.status_code == 200
    assert b'Invalid email address' in response.data  # Assuming the app flashes this message

    # Ensure the original user remains unchanged
    with client.application.app_context():
        unchanged_user = User.query.filter_by(email='john.doe@example.com').first()
        assert unchanged_user is not None
        assert unchanged_user.name == 'John Doe'
        assert unchanged_user.phone == '1234567890'
