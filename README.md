# Face-Recognitio-with-Database-Connectivity
Introduction:
Face recognition is a biometric technology that identifies or verifies an individual's identity by analyzing and comparing facial features from an image or video. It is widely used across industries for security, authentication, and personalization purposes.Google Teachable Machine is a user-friendly, web-based tool that allows anyone to train machine learning models without coding expertise. It is ideal for beginners exploring face recognition.

Objective:
The key objectives of this project are:
To implement a face recognition system using a pre-trained TensorFlow Lite model.
To integrate real-time webcam feeds using OpenCV for dynamic face recognition.
To preprocess input data for compatibility with the pre-trained model and provide accurate predictions.

Prerequisites
Python 3.7 or higher
OpenCV library
NumPy library
A pre-trained face model from Google Teachable Machine

Installation: 1.Visit the Teachable Machine Website: Go to Teachable Machine 
![image](https://github.com/user-attachments/assets/8768ca5a-546a-40cc-b5fa-68a5f13c55a0)

2.Create a New Project: Click on "Get Started" and select "Image Model" under the "New Project" section 3.Select Model Type: Choose the "Standard Image Model" option. 
![image](https://github.com/user-attachments/assets/50707995-59be-4f65-8245-d0cd9ed666f1)

4.Collect Training Data: Use your webcam or upload images to provide examples for each class (e.g., different faces). 
![image](https://github.com/user-attachments/assets/6e248067-8213-41ff-a4c4-d53b2d933ef1)

5.Label Examples: Assign labels to each example 
![image](https://github.com/user-attachments/assets/ffc0e24e-edb6-4166-ae67-040ea9dafe77)

6.Train the Model: Click on the "Train" button to start training your model. 
![image](https://github.com/user-attachments/assets/ebfb7587-a8b3-4a12-b7c7-1065aa62e22f)

7.Export the Model: Once training is complete, click on "Export the Model" and download the model files (a .zip file containing the model weights (.h5) and labels (.txt) files) 
![image](https://github.com/user-attachments/assets/e9a9c6c7-5f73-4fed-9c74-30bb8d17aca1)

#Implementation in Python 1.Set Up Your Environment: Ensure you have Python 3.7 or higher installed. 2.Install Required Libraries: Install OpenCV and NumPy using pip: python ->pip install opencv-python numpy 3.Extract Model Files: Extract the downloaded .h5 and .txt files from the .zip archive and save them in your project directory. 4.Write Python Code: Use the following code to load the model and perform face recognition: from keras.models import load_model # TensorFlow is required for Keras to work import cv2 # Install opencv-python import numpy as np
Install
opencv-python import numpy as np

Disable scientific notation for clarity np.set_printoptions(suppress=True)

Load the model model = load_model("keras_Model.h5", compile=False)

Load the labels class_names = open("labels.txt", "r").readlines()

CAMERA can be 0 or 1 based on default camera of your computer camera = cv2.VideoCapture(0)

while True: # Grab the webcamera's image. ret, image = camera.read()

Resize the raw image into (224-height,224-width) pixels
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

Show the image in a window
cv2.imshow("Webcam Image", image)

Make the image a numpy array and reshape it to the models input shape.
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

Normalize the image array
image = (image / 127.5) - 1

Predicts the model
prediction = model.predict(image) index = np.argmax(prediction) class_name = class_names[index] confidence_score = prediction[0][index]

Print prediction and confidence score
print("Class:", class_name[2:], end="") print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

Listen to the keyboard for presses.
keyboard_input = cv2.waitKey(1)

27 is the ASCII for the esc key on your keyboard.
if keyboard_input == 27: break camera.release() cv2.destroyAllWindows()

#CODE EXPLANATION IMPORTS from keras.models import load_model: Imports the load_model function from Keras to load the pre-trained model. import cv2: Imports the OpenCV library for computer vision tasks. import numpy as np: Imports the NumPy library for numerical operations.

CONFIGURATION np.set_printoptions(suppress=True): Sets the NumPy print options to suppress scientific notation for clarity when printing.

LOAD MODELS AND LABELS model = load_model("keras_Model.h5", compile=False): Loads the pre-trained model from the file keras_Model.h5 without compiling it. class_names = open("labels.txt", "r").readlines(): Reads the class labels from the file labels.txt into a list.

CAMERA SETUP camera = cv2.VideoCapture(0): Opens the default camera (camera index 0) for capturing images.

MAIN LOOPS while True: Starts an infinite loop to continuously capture images. ret, image = camera.read(): Captures an image from the camera. image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA): Resizes the image to 224x224 pixels to match the model's input size. cv2.imshow("Webcam Image", image): Displays the captured image in a window titled "Webcam Image".

PREPROCESS IMAGE image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3): Converts the image to a NumPy array and reshapes it to (1, 224, 224, 3) to match the model's input shape. image = (image / 127.5) - 1: Normalizes the image array to a range of [-1, 1]

MAKE PREDICTION prediction = model.predict(image): Uses the model to predict the class of the input image. index = np.argmax(prediction): Finds the index of the class with the highest confidence score. class_name = class_names[index]: Retrieves the class name corresponding to the predicted index. confidence_score = prediction[0][index]: Retrieves the confidence score of the predicted class.

DISPLAY RESULTS print("Class:", class_name[2:], end=""): Prints the predicted class name. print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%"): Prints the confidence score as a percentage.

HANDLE KEYBOARD INPUT keyboard_input = cv2.waitKey(1): Waits for keyboard input. if keyboard_input == 27:: Checks if the 'Esc' key (ASCII code 27) is pressed to break the loop.

RELEASE RESOURCE camera.release(): Releases the camera resource. cv2.destroyAllWindows(): Closes all OpenCV windows.

#CONCLUSION Using Google Teachable Machine, you can easily create a face recognition model and implement it in Python. This approach is beginner-friendly and customizable, making it a great starting point for learning about machine learning and computer vision.

#ABOUT SQLlite In this project, SQLite (a lightweight, disk-based database) is used to store information about the recognized faces. This integration allows us to keep a record of recognized faces along with some associated data, like the embeddings (feature vectors) of the faces. Here's how SQL is used in various parts of the project:

Database Setup Connecting to the SQLite Database: import sqlite3 conn = sqlite3.connect('face_recognition.db') c = conn.cursor() sqlite3.connect('face_recognition.db'): Connects to the SQLite database file named face_recognition.db. If the file does not exist, it will be created. c = conn.cursor(): Creates a cursor object that allows us to execute SQL commands.

#Creating the Table: c.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, embedding BLOB)''') This SQL command creates a table named faces if it doesn't already exist. The table has three columns: id: An integer primary key that autoincrements for each new record. name: A text field to store the name of the recognized face. embedding: A binary large object (BLOB) to store the face embedding (feature vector).

Inserting Data into the Database Recognizing Faces: def recognize_faces(image): preprocessed_image = preprocess_image(image) prediction = model.predict(preprocessed_image) index = np.argmax(prediction) class_name = class_names[index][2:] confidence_score = prediction[0][index] return class_name, confidence_score This function preprocesses the image, makes a prediction using the model, and returns the class name and confidence score.

Saving Recognized Faces to the Database: def save_to_database(name, embedding): c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, embedding)) conn.commit() c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, embedding)): Executes an SQL command to insert the recognized face's name and embedding into the faces table. The ? placeholders are used to safely insert the values. conn.commit(): Commits the transaction to the database, making the changes permanent.

Main Loop In the main loop, after recognizing a face, we save the recognition result to the database: while True: ret, image = camera.read() if not ret: break class_name, confidence_score = recognize_faces(image) print(f"Class: {class_name}, Confidence Score: {confidence_score * 100:.2f}%") save_to_database(class_name, prediction[0].tobytes()) cv2.imshow("Webcam Image", image) keyboard_input = cv2.waitKey(1) if keyboard_input == 27: # Esc key break

camera.release() cv2.destroyAllWindows() conn.close() save_to_database(class_name, prediction[0].tobytes()): After recognizing a face, we save the class name and the face embedding (converted to bytes) to the database.

#Conclusion Using SQL in this project allows us to maintain a persistent record of recognized faces. This approach can be useful for applications where you need to track recognized faces over time or perform further analysis on the stored data. The integration with SQLite makes the setup simple and portable.
