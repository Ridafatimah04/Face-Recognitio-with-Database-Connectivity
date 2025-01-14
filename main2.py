from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # OpenCV for image processing
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
try:
    model = load_model("C:/Users/Rida Rahil/PycharmProjects/Attendance/keras_model.h5", compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the class labels
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Labels loaded successfully.")
except FileNotFoundError:
    print("Error: 'labels.txt' not found.")
    exit()

# Initialize the webcam
camera = cv2.VideoCapture(0)  # Adjust the index (0, 1, etc.) if you have multiple cameras

if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam initialized. Press ESC to exit.")

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

    # Display prediction and confidence score on the image
    label = f"{class_name}: {np.round(confidence_score * 100, 2)}%"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Print the prediction and confidence score in the console
    print(f"Class: {class_name}, Confidence: {np.round(confidence_score * 100, 2)}%")

    # Exit when ESC is pressed
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # ASCII for ESC key
        print("Exiting...")
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
print("Resources released. Program terminated.")
