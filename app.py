from flask import Flask, request, render_template, make_response, jsonify
import os
import cv2
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config["MAX_CONTENT_LENGTH"] = 10 * 1000 * 1000
app.debug = True

model = load_model("model/inceptionv3_weights.h5")


def preprocess_image(img):
    # Convert the image to a numpy array
    img_array = np.array(img, dtype=np.float32)
    # Preprocess the image by subtracting the mean RGB values and scaling the values to [-1, 1]
    img_array -= np.array([103.939, 116.779, 123.68])
    img_array /= 255.0
    img_array -= 0.5
    img_array *= 2.0
    # Expand the dimensions of the image array to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Define a function to extract frames from a video file
def extract_frames(filename):
    # Open the video file
    cap = cv2.VideoCapture(filename)
    # Read the frames from the video file
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to 299x299 pixels (the input size for the Inception V3 model)
        frame = cv2.resize(frame, (299, 299))
        # Add the frame to the list of frames
        frames.append(frame)
    # Release the video file
    cap.release()
    return frames


# Define a function to detect objects in a list of frames using the Inception V3 model
def detect_objects(frames, query):
    # Preprocess each frame for input to the model
    images = [preprocess_image(frame) for frame in frames]
    # Stack the preprocessed images into a single batch
    images = np.vstack(images)
    # Predict the class probabilities for each image using the model
    preds = model.predict(images)
    # Decode the class probabilities into class labels using the ImageNet labels file
    labels = keras.applications.inception_v3.decode_predictions(preds, top=3)
    # Convert the result to a list of dictionaries with keys 'frame', 'label', and 'score'
    results = []
    for i, label in enumerate(labels):
        if label[0][1] == query:
            result = {'path': f'images/image{i}.png', 'score': label[0][2]}
            results.append(result)
            img = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'images/image{i}.png', img)
    return results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    query = request.form['query']
    # Get the uploaded file from the request
    file = request.files['file']
    # Save the uploaded file to disk
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    # Split the video into frames
    frames = extract_frames(filename)
    # Process the frames using the Google Inception V3 model
    results = detect_objects(frames, query)
    # Render the results page with the search query and object detections
    return render_template('results.html', query=query, results=results)


if __name__ == '__main__':
    app.run(debug=True)
