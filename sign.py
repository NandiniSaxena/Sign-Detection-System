import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import random
import os
import shutil
import cv2
import time
from flask import Flask, render_template, Response, jsonify, send_file

app = Flask(__name__)

# Mapping of digits to sign language gesture descriptions
sign_descriptions = {
    '0': 'Closed fist',
    '1': 'Index finger extended',
    '2': 'Index and middle fingers extended',
    '3': 'Thumb, index, and middle fingers extended',
    '4': 'Four fingers extended, thumb folded',
    '5': 'All fingers extended, palm open',
    '6': 'Thumb and pinky extended, others folded',
    '7': 'Index and middle fingers extended, thumb over others',
    '8': 'Thumb, index, and pinky extended',
    '9': 'Thumb and index finger form a circle, others extended'
}

# Global variables for camera and predictions
cap = None
model = None
last_prediction = "None"
last_description = "None"
confidence = 0.0
frame_count = 0
roi_size = 200
camera_active = False
log_file = None

# Check for GPU availability
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Num GPUs Available: ', len(physical_devices))

# Set working directory and prepare dataset
os.chdir('D:/NewAIML/sign/a/Sign-Language-Digits-Dataset/Dataset')
if os.path.isdir('./train/0/') is False:
    os.mkdir('./train')
    os.mkdir('./test')
    os.mkdir('./valid')

    for i in range(0, 10):
        shutil.move(f'{i}', './train')
        os.mkdir(f'./valid/{i}')
        os.mkdir(f'./test/{i}')

        valid_samples = random.sample(os.listdir(f'./train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'./train/{i}/{j}', f'./valid/{i}')

        test_samples = random.sample(os.listdir(f'./train/{i}'), 5)
        for j in test_samples:
            shutil.move(f'./train/{i}/{j}', f'./test/{i}')

os.chdir('../')

# Define paths
train_path = 'D:/NewAIML/sign/a/Sign-Language-Digits-Dataset/Dataset/train'
valid_path = 'D:/NewAIML/sign/a/Sign-Language-Digits-Dataset/Dataset/valid'
test_path = 'D:/NewAIML/sign/a/Sign-Language-Digits-Dataset/Dataset/test'

# Create data generators
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224, 224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224, 224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224, 224), batch_size=10, shuffle=False)

# Verify dataset sizes
assert train_batches.n == 1712
assert valid_batches.n == 300
assert test_batches.n == 50
assert test_batches.num_classes == valid_batches.num_classes == train_batches.num_classes == 10

# Train or load the model
if os.path.isfile('models/sign_language_model.h5') is False:
    # Load MobileNet
    mobile = tf.keras.applications.mobilenet.MobileNet()

    # Create custom model
    partialModel1 = mobile.layers[-5].output
    partialModel2 = Flatten()(partialModel1)
    output = Dense(units=10, activation='softmax')(partialModel2)
    model = Model(inputs=mobile.input, outputs=output)

    # Freeze layers
    for layer in model.layers[:-23]:
        layer.trainable = False

    # Compile and train
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_batches, validation_data=valid_batches, epochs=10)

    # Save the model
    if os.path.isdir('./models/') is False:
        os.mkdir('./models')
    model.save('models/sign_language_model.h5')
else:
    model = load_model('models/sign_language_model.h5')

# Evaluate model with confusion matrix
test_labels = test_batches.classes
predictions = model.predict(x=test_batches, verbose=0)
cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, without normalization")
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Ensure static/images directory exists
    os.makedirs('static/images', exist_ok=True)
    plt.savefig('static/images/confusion_matrix.png')  # Save to static folder
    plt.close()  # Close the plot to free memory

# Plot confusion matrix
cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# Initialize log file
if not os.path.isdir('./detected_signs/'):
    os.mkdir('./detected_signs')
log_file = open('detection_log.txt', 'a')
log_file.write(f"{'Timestamp':<20} {'Digit':<10} {'Confidence':<12} {'Gesture Description'}\n")
log_file.write("-" * 60 + "\n")

def generate_frames():
    global cap, frame_count, last_prediction, last_description, confidence, camera_active
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        x_start = (w - roi_size) // 2
        y_start = (h - roi_size) // 2

        frame_count += 1
        if frame_count % 5 == 0:
            roi = frame[y_start:y_start + roi_size, x_start:x_start + roi_size]
            img = cv2.resize(roi, (224, 224))
            img_array = np.expand_dims(img, axis=0)
            img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
            predictions = model.predict(img_array, verbose=0)
            predicted_class = class_labels[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100
            last_prediction = predicted_class
            last_description = sign_descriptions[predicted_class]

            if confidence > 80:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.imwrite(f'./detected_signs/sign_{predicted_class}_{int(time.time())}.jpg', frame)
                log_file.write(f"{timestamp:<20} {predicted_class:<10} {confidence:.2f}%      {last_description}\n")
                log_file.flush()

        cv2.rectangle(frame, (x_start, y_start), (x_start + roi_size, y_start + roi_size), (0, 255, 0), 2)
        cv2.putText(frame, f"Sign: {last_prediction} ({confidence:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {last_description}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Optional: Show in OpenCV window
        cv2.imshow('Sign Language Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera_active = False
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global cap, camera_active
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Error: Could not open camera", 500
        camera_active = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global cap, camera_active
    camera_active = False
    if cap is not None:
        cap.release()
        cap = None
    cv2.destroyAllWindows()
    return jsonify({'status': 'Camera stopped'})

@app.route('/predictions')
def get_predictions():
    return jsonify({
        'digit': last_prediction,
        'confidence': f"{confidence:.2f}%",
        'description': last_description
    })

@app.route('/confusion_matrix')
def confusion_matrix():
    return send_file('static/images/confusion_matrix.png')

@app.route('/retrain_model')
def retrain_model():
    global model
    if os.path.isfile('models/sign_language_model.h5'):
        os.remove('models/sign_language_model.h5')
    mobile = tf.keras.applications.mobilenet.MobileNet()
    partialModel1 = mobile.layers[-5].output
    partialModel2 = Flatten()(partialModel1)
    output = Dense(units=10, activation='softmax')(partialModel2)
    model = Model(inputs=mobile.input, outputs=output)
    for layer in model.layers[:-23]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_batches, validation_data=valid_batches, epochs=10)
    model.save('models/sign_language_model.h5')
    return jsonify({'status': 'Model retrained'})

@app.route('/log')
def view_log():
    with open('detection_log.txt', 'r') as f:
        log_content = f.read()
    return Response(log_content, mimetype='text/plain')

if __name__ == '__main__':
    # Ensure static and templates directories exist
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, threaded=True)
