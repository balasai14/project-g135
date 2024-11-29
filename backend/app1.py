import cv2
import face_recognition
from flask import Flask, Response
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load known face encodings and names
with open("backend\\trained_faces.pkl", "rb") as f:
    known_faces, known_names = pickle.load(f)

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video frame.")
            break

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Ensure correct color format

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Compute face encodings
        face_encodings = []
        for face_location in face_locations:
            try:
                encoding = face_recognition.face_encodings(rgb_small_frame, [face_location])[0]
                face_encodings.append(encoding)
            except IndexError:
                print("Error: Unable to encode face.")
                continue

        # Recognize faces
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            if True in matches:
                best_match_index = matches.index(True)
                name = known_names[best_match_index]
            face_names.append(name)

        # Draw boxes and labels
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4  # Scale back up
            right *= 4
            bottom *= 4
            left *= 4

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/multi')
def mutli():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
