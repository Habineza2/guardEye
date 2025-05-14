
from flask import Flask, logging, redirect, render_template, Response, request, url_for
import cv2
import os
import threading
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from deepface import DeepFace
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import pyttsx3
import tensorflow as tf
import requests
from datetime import datetime
import time
import sqlite3
import pickle
from werkzeug.utils import secure_filename
from ultralytics import YOLO










try:
    import RPi.GPIO as GPIO # type: ignore
except ImportError:
    class GPIO:
        BOARD = OUT = IN = HIGH = LOW = BCM = None
        @staticmethod
        def setmode(x): pass
        @staticmethod
        def setup(pin, mode): pass
        @staticmethod
        def output(pin, state): pass
        @staticmethod
        def cleanup(): pass


app = Flask(__name__)






model = load_model('my_model.keras', compile=False)
yolo_model = YOLO("yolov8n.pt")

LOCK_PIN = 18  # Example GPIO pin number
GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK_PIN, GPIO.OUT)
LOCK_PIN = 18  # Example GPIO pin number
GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK_PIN, GPIO.OUT)








#open camera
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)
camera = cv2.VideoCapture(0)
detecting = False
frame_lock = threading.Lock()
frame_to_show = None





def send_email_alert(image_path):
    sender_email = "habinezae73@gmail.com"
    receiver_email = "your_email@example.com"
    password = "ihda bqgy macg rebk"
    msg = MIMEMultipart()
    msg['subject'] = 'Thief Detected Alert'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    body = MIMEText(f"A potential thief was detected. Image saved as {os.path.basename(image_path)}")
    msg.attach(body)
    with open(image_path, 'rb') as img_file:
        msg.attach(MIMEImage(img_file.read(), name=os.path.basename(image_path)))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            print("Email sent.")
    except Exception as e:
        print(f"Failed to send email: {e}")






def analyze_emotion(frame):
    try:
        temp_filename = "temp_frame.jpg"
        cv2.imwrite(temp_filename, frame)
        analysis = DeepFace.analyze(temp_filename, actions=['emotion'], enforce_detection=False)
        return analysis
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        return None





def analyze_thief(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unreadable")

        img = cv2.resize(img, (150, 150))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0

        prediction = model.predict(img)

        # For binary classification (1 = Thief, 0 = Not Thief)
        is_thief = prediction[0][0] >= 0.5

        print(f"[Thief Detection] Prediction: {prediction[0][0]:.2f} - {'Thief' if is_thief else 'Not Thief'}")
        return is_thief

    except Exception as e:
        print(f"[ERROR] Thief detection failed: {e}")
        return False





def speak_warning():
    engine = pyttsx3.init()
    engine.say("Warning! You are too close. Please step back or I shut.")
    engine.runAndWait()





def load_known_faces(db_path='users.db'):
    encodings, names = [], []

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute("""
            SELECT u.name, e.encoding
            FROM users u
            JOIN face_encodings e ON u.id = e.user_id
        """)

        for name, encoding_blob in c.fetchall():
            encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            encodings.append(encoding)
            names.append(name)

    except Exception as e:
        print(f"[ERROR] Failed to load known faces: {e}")
    finally:
        conn.close()

    return encodings, names








def log_intrusion(image_path, emotion, is_thief):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO intrusions (timestamp, image_path, emotion, is_thief) VALUES (?, ?, ?, ?)",
          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path, emotion, int(is_thief)))

    conn.commit()
    conn.close()







# Open door

def open_lock():
    GPIO.output(LOCK_PIN, GPIO.HIGH)  # Unlock
    print("Door unlocked.")
    time.sleep(5)  # Keep open for 5 seconds
    GPIO.output(LOCK_PIN, GPIO.LOW)   # Lock again
    print("Door locked.")









def register_user(name, email, phone, image_paths):
    if len(image_paths) != 5:
        print("❌ Please provide exactly 5 image paths.")
        return

    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    try:
        # Insert user details into users table
        c.execute("INSERT INTO users (name, email, phone) VALUES (?, ?, ?)", (name, email, phone))
        user_id = c.lastrowid

        encodings_stored = 0
        for image_path in image_paths:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 0:
                print(f"⚠️ No face found in {image_path}. Skipping.")
                continue
            
            for encoding in encodings:
              encoding_blob = encoding.tobytes()
              c.execute("INSERT INTO face_encodings (user_id, encoding) VALUES (?, ?)", (user_id, encoding_blob))
              encodings_stored += 1

        if encodings_stored == 0:
            print("❌ No valid face encodings found. Registration failed.")
            c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        else:
            conn.commit()
            print(f"✅ User '{name}' registered with {encodings_stored} encodings.")

    except sqlite3.IntegrityError as e:
        print(f"❌ Registration failed: {e}")
    finally:
        conn.close()











def detect_faces():
    global frame_to_show, detecting

    known_encodings, known_names = load_known_faces()

    image_count = 0
    last_save_time = 0
    save_interval = 10  # seconds
    threshold = 0.2
    frame_count = 0

    while detecting:
        success, full_frame = camera.read()
        if not success:
            continue

        frame_count += 1
        display_frame = full_frame.copy()

        # Crop area of interest
        y1, y2 = 100, 400
        x1, x2 = 150, 500
        detection_frame = full_frame[y1:y2, x1:x2]
        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)

        try:
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        except Exception as e:
            print(f"[ERROR] Face detection error: {e}")
            continue

        if face_locations:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=threshold)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1

                name = "Unknown"
                if best_match_index != -1 and matches[best_match_index]:
                    name = known_names[best_match_index]
                    image_path = f"{output_dir}/{name}_{image_count}.jpg"
                    cv2.imwrite(image_path, detection_frame)
                    open_lock()
                    print(f"[INFO] Known person detected: {name}")
                else:
                    now = time.time()
                    if now - last_save_time > save_interval:
                        image_count += 1
                        image_path = f"{output_dir}/Unknown_{image_count}.jpg"
                        print("[INFO] Unknown person detected")
                        try:
                            cv2.imwrite(image_path, detection_frame)
                            emotion_result = analyze_emotion(detection_frame)
                            thief = analyze_thief(image_path)
                            dominant_emotion = emotion_result[0]['dominant_emotion'] if emotion_result else "Unknown"

                            if thief:
                                send_email_alert(image_path)
                            log_intrusion(image_path, dominant_emotion, thief)

                            if emotion_result:
                                print(f"[INFO] Emotion detected: {dominant_emotion}")
                        except Exception as e:
                            print(f"[ERROR] Error processing unknown face: {e}")
                        last_save_time = now

                # Adjust coordinates to full frame
                top += y1
                bottom += y1
                left += x1
                right += x1
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(display_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        else:
            print("[WARNING] No face detected. Running YOLO for suspicious activity...")
            if frame_count % 3 == 0:
                try:
                    results = yolo_model(full_frame, conf=0.25)
                    suspicious, persons = detect_suspicious_activity(full_frame, results[0], face_locations)
                    if suspicious:
                        for box in persons:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(display_frame, "Suspicious", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print("[ALERT] Suspicious person detected hiding their face.")
                except Exception as e:
                    print(f"[ERROR] YOLO failed: {e}")

        with frame_lock:
            frame_to_show = display_frame.copy()




def initialize_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    
    c.execute("PRAGMA foreign_keys = ON")

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            phone TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            encoding BLOB NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS intrusions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            emotion TEXT,
            is_thief INTEGER,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()

initialize_db()















API_KEY = "d95d9c184b6714fc105726523e9c168a"  
CITY = "Musanze"  

def fetch_weather_data(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        return temperature, humidity
    else:
        return None, None








@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        images = request.files.getlist("images")
        


        if len(images) != 5:
            return "Please upload exactly 5 images."

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()

            # Insert user basic info into 'users' table
            image_paths = []
            for image in images:
                image_filename = secure_filename(image.filename)
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)
                image_paths.append(image_path)

            # Insert user details and image path
            c.execute('''
                INSERT INTO users (name, email, phone)
                VALUES (?, ?, ?)
            ''', (name, email, phone))  # Save the path of the first image, or handle the logic accordingly.
            
            user_id = c.lastrowid  # get user ID for linking images

            for image_path in image_paths:
                image_loaded = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image_loaded)

                if not encodings:
                    conn.close()
                    return f"No face detected in {image_filename}. Please upload clear face images."

                face_encoding = encodings[0]
                encoding_blob = face_encoding.tobytes()

                # Insert encoding into face_encodings table
                c.execute('''
                    INSERT INTO face_encodings (user_id, image_path, encoding)
                    VALUES (?, ?, ?)
                ''', (user_id, image_path, encoding_blob))

            conn.commit()
            conn.close()

            # Fetch environment data
            temp, hum = fetch_weather_data(CITY)
            data = {
                "temperature": temp or 0,
                "humidity": hum or 0,
                "fire_status": "No Fire",
                "last_entry": datetime.now().strftime("%b %d, %H:%M"),
                "temp_history": [24.1, 24.2, 24.3, 24.1, 24.4, 10.0, 24.3, 24.2, 10.5, 24.3]
            }

            return render_template('dashboard.html', data=data)

        except Exception as e:
            print(f"[ERROR] Failed to register user: {e}")
            return "Registration failed. Check logs for details."

    return render_template('signup.html')











@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('view_users')) 









@app.route('/')
def index():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')


@app.route("/dashboard")
def dashboard():
    temp, hum = fetch_weather_data(CITY)
    print(temp, hum)
    data = {
        "temperature": temp or 0,
        "humidity": hum or 0,
        "fire_status": "No Fire", 
        "last_entry": datetime.now().strftime("%b %d, %H:%M"),
        "temp_history": [24.1, 24.2, 24.3, 24.1, 24.4, 10.0, 24.3, 24.2, 10.5, 24.3]
    }
    return render_template("dashboard.html", data=data)








@app.route('/users')
def view_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, name, email, phone FROM users")
    users = c.fetchall()
    conn.close()
    return render_template('users.html', users=users)






@app.route('/intrusions')
def view_intrusions():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, timestamp, image_path, emotion, is_thief FROM intrusions ORDER BY timestamp DESC" )
    logs = c.fetchall()
    conn.close()
    return render_template('intrusions.html', logs=logs)








@app.route('/start')
def start():
    global detecting
    if not detecting:
        detecting = True
        threading.Thread(target=detect_faces, daemon=True).start()
    return "Detection started"

@app.route('/stop')
def stop():
    global detecting
    detecting = False
    return "Detection stopped"

def gen_frames():
    global frame_to_show
    while True:
        with frame_lock:
            if frame_to_show is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame_to_show)
            frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
