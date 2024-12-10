import time
import webbrowser
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, session
from PIL import Image, ImageOps
import base64
from io import BytesIO
import requests
import face_recognition
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management
CORS(app)

# Global variable for video capture object (initialized when needed)
video_capture = None

# Global variable for reference face encoding (initially set to None)
reference_face_encoding = None
global_intId = None

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to handle image orientation based on EXIF data
def handle_exif_orientation(image):
    try:
        image = ImageOps.exif_transpose(image)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

# Function to fetch reference image URL dynamically using intId
def get_reference_image_url(intId):
    url = "https://ssemeyapi.softsolanalytics.com/API/AcademicClassSchedular/GetImageByStudentId"
    payload = {"intId": intId}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        if response.status_code == 200:
            data = response.json()
            print(data)

            data_list = data.get('data')
            if isinstance(data_list, list) and len(data_list) > 0:
                str_file_name = data_list[0].get('strFileName')
                if str_file_name:
                    return str_file_name.strip('"')
                else:
                    print("The 'strFileName' key is missing or null in the first item of 'data'.")
                    return ""
            else:
                print("The 'data' key is missing or not a list in the API response.")
                return ""
        else:
            print(f"API request failed with status code {response.status_code}")
            return ""
    except requests.exceptions.RequestException as e:
        return ""

# Function to load and encode reference image
def load_reference_image(reference_image_url=None, intId=None):
    global reference_face_encoding

    try:
        if not reference_image_url:
            reference_image_url = get_reference_image_url(intId)
        print(reference_image_url)
        if not reference_image_url:
            raise ValueError("Reference image URL could not be determined.")

        response = requests.get(reference_image_url)
        response.raise_for_status()

        reference_image = Image.open(BytesIO(response.content))

        if reference_image.mode != 'RGB':
            reference_image = reference_image.convert('RGB')

        reference_image = handle_exif_orientation(reference_image)
        reference_image_np = np.array(reference_image)

        reference_face_encodings = face_recognition.face_encodings(reference_image_np, model="hog")

        if len(reference_face_encodings) == 0:
            raise ValueError("No faces found in the reference image.")
        
        reference_face_encoding = reference_face_encodings[0]
    except Exception as e:
        raise ValueError(f"Error loading reference image: {str(e)}")

# Default route with intId handling
@app.route('/')
def index():
    global global_intId
    intId = request.args.get('intId')
    global_intId = int(intId)
    if not intId:
        return jsonify({'message': 'intId is required in the query string'}), 400

    try:
        reference_image_url = f"https://videoconferencing.softsolanalytics.com/vdc/StudentImage/{int(intId)}.jpeg"
        load_reference_image(reference_image_url=reference_image_url, intId=intId)

        return render_template('index.html', intId=intId)
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Compare images (existing functionality)
def compare_images(new_image, reference_face_encoding, threshold=0.35):
    if new_image.mode != 'RGB':
        new_image = new_image.convert('RGB')

    new_image = new_image.resize((new_image.width // 2, new_image.height // 2))

    face_locations = face_recognition.face_locations(np.array(new_image), model="hog")
    if len(face_locations) == 0:
        return "No faces."

    face_encodings = face_recognition.face_encodings(np.array(new_image), model="hog")
    if len(face_encodings) == 0:
        return "No faces found in the new image."

    if len(face_encodings) > 1:
        return f"Multiple faces detected. Only the first face will be used for comparison."

    new_face_encoding = face_encodings[0]
    face_distance = face_recognition.face_distance([reference_face_encoding], new_face_encoding)[0]

    if face_distance < threshold:
        print("The new image matches the reference image!")
        return 1

    else:
        print("The new image does not match the reference image.")
        return 0

# Video stream route using OpenCV with face detection logic and face mesh
@app.route('/video_feed')
def video_feed():
    
    global video_capture, reference_face_encoding

    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)

    face_detected = False
    start_time = time.time()

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the session flag at the beginning of video feed handling
    session['redirect_to_response'] = False  # Default value

    def generate():
        nonlocal face_detected, start_time
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                face_detected = True

                # For each face detected, apply face mesh using MediaPipe
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_region = frame[y:y + h, x:x + w]
                    rgb_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

                    # Process face region with MediaPipe face mesh
                    results = face_mesh.process(rgb_face_region)

                    if results.multi_face_landmarks:
                        for landmarks in results.multi_face_landmarks:
                            for landmark in landmarks.landmark:
                                # Draw landmarks
                                cx, cy = int(landmark.x * w) + x, int(landmark.y * h) + y
                                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

            # Convert frame to JPEG for streaming
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            # Yield frame as a multipart response (for streaming)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Stop the video feed after 3 seconds, unless a face is detected
            if time.time() - start_time > 5 and not face_detected:
                break
            elif face_detected:
                time.sleep(5)  # Let the feed continue briefly after detection
                break

        # After the video feed ends, check if a face was detected
        if face_detected:
            print("Face detected, moving to face recognition...")

            # Proceed to face recognition with the current frame
            img = Image.fromarray(frame)
            result = compare_images(img, reference_face_encoding)
            print(result)

            if result == 0:
                global global_intId
                if get_reference_image_url(global_intId) != "":
                 #   get_reference_image_url(global_intId)
                    reference_image_url=get_reference_image_url(global_intId)
                    load_reference_image(reference_image_url, intId=global_intId)
                    
                result = compare_images(img, reference_face_encoding)
            if result == 1:
             return webbrowser.open_new_tab('http://127.0.0.1:5500/ResponseData.html') # chnage url 
            else:
               return webbrowser.open_new_tab('http://127.0.0.1:5500/NoResponse.html') # chnage url 
        else:
            print("No faces found.")
            result = "No faces found."
            session['redirect_to_response'] = False

    # Return video feed as a respoanse (MIME type for multipart)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)
