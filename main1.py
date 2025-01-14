from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # OpenCV for image processing
import numpy as np
import mysql.connector  # MySQL database connection
import os

# Disable GPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Database connection
try:
    db_connection = mysql.connector.connect(
        host="localhost",  # Replace with your MySQL server address
        user="root",       # Replace with your MySQL username
        password="Rida._.7879",  # Replace with your MySQL password
        database="attendance"  # Replace with your database name
    )
    print("Database connected successfully.")
except mysql.connector.Error as e:
    print(f"Error connecting to the database: {e}")
    exit()

# Load the model
try:
    model = load_model(r"C:/Users/Rida Rahil/PycharmProjects/Attendance/keras_model.h5", compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    db_connection.close()
    exit()

# Load the class labels
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Labels loaded successfully: {class_names}")
except FileNotFoundError:
    print("Error: 'labels.txt' not found.")
    db_connection.close()
    exit()

# Initialize the webcam
camera = cv2.VideoCapture(0)  # Adjust the index (0, 1, etc.) if you have multiple cameras

if not camera.isOpened():
    print("Error: Could not access the webcam.")
    db_connection.close()
    exit()

print("Webcam initialized. Press ESC to exit.")

# Dictionary to track attendance (to avoid duplicate entries)
attendance_marked = {}

try:
    # Loop for real-time predictions
    while True:
        # Capture the webcam image
        ret, image = camera.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break

        # Resize the image to the model's input size (224x224 pixels)
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Convert the image to a numpy array and normalize it
        image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1  # Normalize to range [-1, 1]

        # Perform prediction
        try:
            prediction = model.predict(image_array, verbose=0)  # Suppress verbose output
            index = np.argmax(prediction)  # Get the class index with the highest probability
            class_name = class_names[index]  # Get the class label
            confidence_score = prediction[0][index]  # Confidence score for the predicted class
        except Exception as e:
            print(f"Error during prediction: {e}")
            break

        # Clean the class name (handle numeric prefixes or unwanted characters)
        class_name_cleaned = class_name.split(" ", 1)[-1].strip()

        # Check if confidence score is above 50%
        if confidence_score > 0.5:
            # Display prediction and confidence score on the image
            label = f"{class_name_cleaned}: {np.round(confidence_score * 100, 2)}%"
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Check if the person has already been marked present
            if class_name_cleaned not in attendance_marked:
                try:
                    # Query the database for the person's details
                    cursor = db_connection.cursor(buffered=True)
                    cursor.execute("SELECT id, name, register_number FROM persons WHERE name = %s", (class_name_cleaned,))
                    result = cursor.fetchone()

                    if result:
                        person_id, person_name, reg_no = result
                        print(f"Marking attendance for {person_name} (ID: {person_id}, Reg No: {reg_no})")

                        # Update attendance record in the database
                        cursor.execute("UPDATE persons SET attendance = 1 WHERE id = %s", (person_id,))
                        db_connection.commit()

                        # Mark the person as attended in the session
                        attendance_marked[class_name_cleaned] = True
                        print(f"Attendance marked for {person_name}.")
                    else:
                        print(f"Person '{class_name_cleaned}' not found in the database.")

                except mysql.connector.Error as db_error:
                    print(f"Database error: {db_error}")

                finally:
                    cursor.close()  # Close the cursor

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Print the prediction and confidence score in the console
        print(f"Class: {class_name_cleaned}, Confidence: {np.round(confidence_score * 100, 2)}%")

        # Exit when ESC is pressed
        keyboard_input = cv2.waitKey(1)
        if keyboard_input == 27:  # ASCII for ESC key
            print("Exiting...")
            break

except Exception as main_exception:
    print(f"Unexpected error occurred: {main_exception}")

finally:
    # Release resources
    camera.release()
    cv2.destroyAllWindows()
    db_connection.close()
    print("Resources released. Program terminated.")

