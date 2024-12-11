import re
from datetime import timedelta
from datetime import datetime
from flask import Flask
from flask import render_template, redirect, url_for
from flask import request
from flask import session
from flask import flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate

from sqlalchemy import and_


# from flask import app

app = Flask(__name__)


app.permanent_session_lifetime = timedelta(minutes=30)

app.secret_key = "SECRET_KEY"

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://sheri:sheri123@localhost/smart_underage_driver_detector'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)


# Define the User model
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(200), nullable=False)


# Define the Location model
class Location(db.Model):
    __tablename__ = "locations"
    id = db.Column(db.Integer, primary_key=True)
    location_name = db.Column(db.String(200), nullable=False)


# Define the Result model
class Result(db.Model):
    __tablename__ = "results"
    id = db.Column(db.Integer, primary_key=True)
    location_id = db.Column(db.Integer, db.ForeignKey("locations.id"), nullable=False)
    date_time = db.Column(db.DateTime, nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    location = db.relationship("Location", backref=db.backref("results", lazy=True))
    underage_status = db.Column(db.String(50), nullable=False)


# Create a Flask application instance
def create_app():
    app = Flask(__name__)

    # Load the configuration from the given config class
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://sheri:sheri123@localhost/smart_underage_driver_detector'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.secret_key = 'Secret_key'
    app.permanent_session_lifetime = timedelta(minutes=30)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # Return the app instance for use in tests or the main program
    return app

# Create the database tables
with app.app_context():
    db.create_all()


# Home route
@app.route("/")
def home():
    return render_template("index.html")


# Error handlers for better UX
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500


# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    # session.pop("_flashes", None)
    if request.method == "POST":
        email = request.form.get("email").strip()
        password = request.form.get("password")

        # Basic email validation
        if not email or not password:
            flash("Please provide both email and password.", "error")
            return redirect(url_for("login"))

        # Fetch the user from the database
        user = User.query.filter_by(email=email).first()

        # Check if the user exists and password matches
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["email"] = user.email
            session["logged_in"] = True
            session.permanent = True  # Set session timeout
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials! Please try again.", "error")
            return redirect(url_for("login"))

    return render_template("login.html")


# Signup route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]

        phone_regex = r"^\d{11}$"  # 11 digit phone number
        password_regex = r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$"

        # Check if the user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("User already exists!", "error")
            return render_template("signup.html", name=name, email=email, phone=phone)

        # Phone validation
        if not re.match(phone_regex, phone):
            flash("Invalid phone number! It should be 11 digits long.", "error")
            return render_template("signup.html", name=name, email=email, phone=phone)

        # Password validation
        if not re.match(password_regex, password):
            flash(
                "Password must be at least 8 characters long and include both letters and numbers.",
                "error",
            )
            return render_template("signup.html", name=name, email=email, phone=phone)

        # Create a new user if validation pass
        new_user = User(
            name=name,
            email=email,
            phone=phone,
            password=generate_password_hash(password),
        )
        db.session.add(new_user)
        db.session.commit()

        session["user_id"] = new_user.id
        session["logged_in"] = True

        flash("Account created successfully!", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


# Update Profile route
@app.route("/update_profile", methods=["GET", "POST"])
def update_profile():
    # Check if user is logged in
    if "user_id" not in session:
        return redirect(url_for("login"))
    session.pop("_flashes", None)
    # Fetch the user based on user_id in the session
    user = User.query.get(session["user_id"])

    if request.method == "POST":
        # Get updated information from the form
        user.name = request.form["name"]
        user.email = request.form["email"]
        user.phone = request.form["phone"]

        # Optionally, update password if provided
        if request.form["password"]:  # Check if the password field is not empty
            user.password = generate_password_hash(request.form["password"])

        # Commit changes to the database
        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("dashboard"))

    # Render the update profile form with current user data
    return render_template("update_profile.html", user=user)


# Dashboard route
@app.route("/dashboard")
def dashboard():
    # Check if the user is logged in by verifying session
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Query the user based on the stored user_id in session
    user = User.query.filter_by(id=session["user_id"]).first()

    if not user:
        flash("User not found. Please log in again.", "error")
        return redirect(url_for("login"))

    # Retrieve locations from the database (assuming you have a Location model)
    locations = Location.query.all()

    # Pass the user's name and locations to the dashboard template
    return render_template("dashboard.html", name=user.name, locations=locations)


# Add location route
@app.route("/add_location", methods=["GET", "POST"])
def add_location():
    if "email" not in session:
        return redirect(url_for("login"))

    # session.pop("_flashes", None)
    locations = Location.query.all()  # Fetch all locations for display if needed

    if request.method == "POST":
        location_name = request.form["location"].strip()  # Trim extra spaces

        # Input validation
        if not location_name:
            flash("Location is required!", "error")
        elif len(location_name) < 5 or len(location_name) > 255:
            flash("Location must be between 5 and 255 characters.", "error")
        elif not re.match(r"^[a-zA-Z0-9\s,.'-]{5,255}$", location_name):
            flash("Location contains invalid characters.", "error")
        else:
            # Check if the location already exists
            existing_location = Location.query.filter_by(location_name=location_name).first()
            if existing_location:
                flash("Location already exists!", "error")
            else:
                try:
                    # Add new location to the database
                    new_location = Location(location_name=location_name)
                    db.session.add(new_location)
                    db.session.commit()
                    flash("Location added successfully!", "success")
                except Exception as e:
                    db.session.rollback()
                    flash(f"An error occurred while adding the location: {str(e)}", "error")

        return redirect(url_for("add_location"))

    return render_template("add_location.html", locations=locations)


# Delete location route
@app.route("/delete_location", methods=["GET", "POST"])
def delete_location():
    if "email" not in session:
        return redirect(url_for("login"))

    # session.pop("_flashes", None)
    locations = Location.query.all()

    if request.method == "POST":
        location_id = request.form["location_id"]
        location_to_delete = Location.query.get(location_id)

        if location_to_delete:
            db.session.delete(location_to_delete)
            db.session.commit()
            flash("Location deleted successfully!", "success")
        else:
            flash("Location not found!", "error")
        return redirect(url_for("delete_location"))

    return render_template("delete_location.html", locations=locations)


@app.route("/start_capture", methods=["POST"])
def start_capture():
    try:
        # Import necessary libraries
        import cv2
        import os
        import numpy as np
        from tensorflow.keras.models import load_model
        from datetime import datetime

        # Fetch the selected location dynamically from the form
        selected_location_id = request.form.get("location_id", None)

        if not selected_location_id:
            flash("No location selected. Please select a location.", "error")
            return redirect(url_for("dashboard"))

        # Get the selected location from the database
        selected_location = Location.query.get(selected_location_id)
        if not selected_location:
            flash("Invalid location selected.", "error")
            return redirect(url_for("dashboard"))

        # Load YOLO configuration and weights
        yolo_weights = "./yolov3.weights"
        yolo_cfg = "./yolov3.cfg"
        coco_names = "./coco.names"

        # Load YOLO model
        net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Load classes
        with open(coco_names, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Load the trained model for age prediction
        loaded_model = load_model("models/age_prediction_sigmoid256_modelkeras.keras")

        # Output directory for saving faces
        output_folder = "static/result"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Image dimensions for preprocessing
        image_width, image_height = 256, 256

        def preprocess_image(image):
            img = cv2.resize(image, (image_width, image_height))  # Resize to 256x256
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = img / 255.0  # Normalize pixel values
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img

        # Start webcam capture
        cap = cv2.VideoCapture(0)  # 0 for default camera
        print("Press 'c' to capture an image and process it, or 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            cv2.imshow("Press 'c' to Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):  # Capture frame on 'c' key
                height, width = frame.shape[:2]

                # Prepare the image for YOLO
                blob = cv2.dnn.blobFromImage(
                    frame, 0.00392, (608, 608), (0, 0, 0), True, crop=False
                )
                net.setInput(blob)
                outputs = net.forward(output_layers)

                faces = []
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if class_id == 0 and confidence > 0.4:  # Detect faces
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = max(0, int(center_x - w / 2))
                            y = max(0, int(center_y - h / 2))
                            faces.append((x, y, w, h))

                # If multiple faces, choose the leftmost
                if faces:
                    leftmost_face = min(
                        faces, key=lambda b: b[0]
                    )  # Select face with smallest x-coordinate
                    x, y, w, h = leftmost_face
                    face = frame[y : y + h, x : x + w]

                    if face.size > 0:
                        # Preprocess the face for the model
                        preprocessed_face = preprocess_image(face)

                        # Predict age
                        prediction = loaded_model.predict(preprocessed_face)[0][0]
                        predicted_label = int(round(prediction))  # 0 or 1
                        label_text = (
                            "underage" if predicted_label == 0 else "notunderage"
                        )

                        # Save face with label
                        save_name = f"captured_{label_text}_{len(os.listdir(output_folder)) + 1}.jpg"
                        save_path = os.path.join(output_folder, save_name)
                        resized_face = cv2.resize(face, (image_width, image_height))
                        cv2.imwrite(save_path, resized_face)

                        print(f"Saved labeled face: {save_path}")

                        # Add result to database only if underage
                        try:
                            # if label_text == "underage":
                                # Create a new result record
                                new_result = Result(
                                    location_id=selected_location.id,
                                    date_time=datetime.now(),
                                    image_path=save_path,
                                    underage_status=label_text,
                                )
                                db.session.add(new_result)
                                db.session.commit()
                                print("Underage result saved to database.")
                            # else:
                            #     print(
                            #         "Prediction is 'notunderage'; skipping database save."
                            #     )
                        except Exception as db_error:
                            db.session.rollback()
                            print(f"Failed to save result to database: {db_error}")
                else:
                    print("No faces detected. Try again.")

            elif key == ord("q"):  # Quit on 'q' key
                print("Exiting...")
                break

        cap.release()
        cv2.destroyAllWindows()

        # Flash success message and redirect back to dashboard
        flash("Capture started successfully!", "success")
        return redirect(url_for("dashboard"))

    except Exception as e:
        # Flash error message if any exception occurs
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for("dashboard"))


@app.route("/show_results", methods=["GET", "POST"])
def show_results():
    session.pop("_flashes", None)
    if "email" not in session:
        return redirect(url_for("login"))

    locations = Location.query.all()
    location_filter = request.args.get("location", None)
    start_date = request.args.get("start_date", None)
    end_date = request.args.get("end_date", None)

    query = Result.query.join(Location).add_columns(
        Result.id, Result.date_time, Result.image_path, Location.location_name
    )

    # Apply location filter
    if location_filter:
        query = query.filter(Location.location_name == location_filter)

    # Apply date filters
    try:
        if start_date and end_date:
            # Parse both dates
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            # Apply date range filter
            query = query.filter(
                and_(Result.date_time >= start, Result.date_time <= end)
            )
        elif start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(Result.date_time >= start)
        elif end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
            query = query.filter(Result.date_time <= end)
    except ValueError:
        flash("Invalid date format. Please use YYYY-MM-DD.", "error")
        return redirect(url_for("show_results"))

    # Debugging
    print(f"Location Filter: {location_filter}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print("Generated Query:", str(query))

    page = request.args.get("page", 1, type=int)
    per_page = 10
    pagination = query.paginate(page=page, per_page=per_page)

    filters_applied = any([location_filter, start_date, end_date])

    return render_template(
        "show_results.html",
        results=pagination.items,
        pagination=pagination,
        location_filter=location_filter,
        start_date=start_date,
        end_date=end_date,
        locations=locations,
        filters_applied=filters_applied,
    )


# Logout route
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# Main entry point
if __name__ == "__main__":
    app.run(debug=True)
