import cv2
import time
import threading
import os
import shutil
import pickle
import numpy as np
import face_recognition
import mysql.connector
from flask import Flask, Response, jsonify, request
from config import db_config

# end-to-end workflow for eyespy face detection
# Unknown visitor → session → images saved → user labels → encodings stored → future recognition

app = Flask(__name__)

camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

state = {
    "face_detected": False,
    "face_count": 0,
    "last_seen": None,
    "fps": 0
}

lock = threading.Lock()

session = {
    "active": False,
    "id": None,
    "path": None,
    "start_time": None,
    "last_seen": None,
    "image_count": 0,
    "last_capture_time": 0
}

BASE_PATH = "./data/snapshots/unknown"
FACES_PATH = "./data/faces"
ENCODINGS_PATH = "./data/encodings.pkl"

MIN_FACE_SIZE = 80
STABILITY_TIME = 1
SESSION_TIMEOUT = 3
MIN_IMAGES = 3
MAX_IMAGES = 5
MAX_ENCODINGS_PER_LABEL = 20
TOLERANCE = 0.6
RECOGNITION_INTERVAL = 1.0

known_encodings = []
known_labels = []
encoding_store = {}

last_recognition_time = 0
last_recognition_results = []

def load_encodings():
    global encoding_store
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            encoding_store = pickle.load(f)
    else:
        encoding_store = {}
    rebuild_memory()

def save_encodings():
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(encoding_store, f)

def rebuild_memory():
    global known_encodings, known_labels
    known_encodings = []
    known_labels = []
    for label in encoding_store:
        for enc in encoding_store[label]:
            known_encodings.append(enc)
            known_labels.append(label)

def add_encoding(label, enc):
    if label not in encoding_store:
        encoding_store[label] = []
    encoding_store[label].append(enc)
    if len(encoding_store[label]) > MAX_ENCODINGS_PER_LABEL:
        encoding_store[label].pop(0)

def get_db_connection():
    return mysql.connector.connect(**db_config)

def create_session_record(session_id, path, start_time):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (session_id, path, start_time, status) VALUES (%s, %s, FROM_UNIXTIME(%s), 'unknown')",
        (session_id, path, start_time)
    )
    conn.commit()
    cursor.close()
    conn.close()

def end_session_record(session_id, end_time):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sessions SET end_time = FROM_UNIXTIME(%s) WHERE session_id = %s",
        (end_time, session_id)
    )
    conn.commit()
    cursor.close()
    conn.close()

def update_session_label(session_id, label):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sessions SET status = 'known', label = %s WHERE session_id = %s",
        (label, session_id)
    )
    conn.commit()
    cursor.close()
    conn.close()

def create_session():
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_id = "unknown_" + ts
    path = os.path.join(BASE_PATH, session_id)
    os.makedirs(path, exist_ok=True)
    session["active"] = True
    session["id"] = session_id
    session["path"] = path
    session["start_time"] = time.time()
    session["last_seen"] = time.time()
    session["image_count"] = 0
    session["last_capture_time"] = 0
    create_session_record(session_id, path, session["start_time"])

def end_session():
    if session["id"]:
        if session["image_count"] >= MIN_IMAGES:
            end_session_record(session["id"], time.time())
    session["active"] = False
    session["id"] = None
    session["path"] = None
    session["start_time"] = None
    session["last_seen"] = None
    session["image_count"] = 0
    session["last_capture_time"] = 0

def save_snapshot(frame):
    if session["image_count"] >= MAX_IMAGES:
        return
    now = time.time()
    if now - session["last_capture_time"] < 1:
        return
    session["last_capture_time"] = now
    session["image_count"] += 1
    filename = str(session["image_count"]) + ".jpg"
    path = os.path.join(session["path"], filename)
    cv2.imwrite(path, frame)

def recognise_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    results = []

    for (top, right, bottom, left), face_enc in zip(locations, encodings):
        matches = face_recognition.compare_faces(known_encodings, face_enc, TOLERANCE)
        label = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            label = known_labels[matched_idx]

        results.append((top, right, bottom, left, label))

    return results

def process_frame(frame):
    global last_recognition_time, last_recognition_results, face_detect_start_time

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    valid_faces = [(x, y, w, h) for (x, y, w, h) in faces if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE]

    now = time.time()
    face_detected = len(valid_faces) > 0

    with lock:
        state["face_count"] = len(valid_faces)
        state["face_detected"] = face_detected
        if face_detected:
            state["last_seen"] = now

    if face_detected:
        if face_detect_start_time is None:
            face_detect_start_time = now
    else:
        face_detect_start_time = None

    stable_face = face_detect_start_time and (now - face_detect_start_time >= STABILITY_TIME)

    if stable_face and not session["active"]:
        create_session()

    if stable_face and session["active"]:
        session["last_seen"] = now
        save_snapshot(frame)

    if not face_detected and session["active"]:
        if now - session["last_seen"] > SESSION_TIMEOUT:
            end_session()

    if now - last_recognition_time >= RECOGNITION_INTERVAL:
        last_recognition_results = recognise_face(frame)
        last_recognition_time = now

    for (top, right, bottom, left, label) in last_recognition_results:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        start = time.time()
        frame = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        end = time.time()

        with lock:
            state["fps"] = 1 / (end - start if end - start > 0 else 1)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/label', methods=['POST'])
def label_session():
    data = request.json
    session_id = data.get("session_id")
    label = data.get("label")

    source_path = os.path.join(BASE_PATH, session_id)
    target_path = os.path.join(FACES_PATH, label)

    if not os.path.exists(source_path):
        return jsonify({"error": "session not found"}), 404

    os.makedirs(target_path, exist_ok=True)

    for filename in os.listdir(source_path):
        src = os.path.join(source_path, filename)
        dst = os.path.join(target_path, session_id + "_" + filename)
        shutil.copy(src, dst)

        image = face_recognition.load_image_file(src)
        encs = face_recognition.face_encodings(image)

        for enc in encs:
            add_encoding(label, enc)

    save_encodings()
    rebuild_memory()
    update_session_label(session_id, label)

    return jsonify({"status": "labelled", "label": label})

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    with lock:
        return jsonify(state)

@app.route('/')
def index():
    return "EyeSpy running"

if __name__ == '__main__':
    face_detect_start_time = None
    load_encodings()
    app.run(host='0.0.0.0', port=5000, debug=False)