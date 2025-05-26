import json
from flask import Flask, flash, json, logging, redirect, render_template, Response, request, session, url_for
import cv2
from werkzeug.security import generate_password_hash, check_password_hash
from flask import send_file
import io
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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import h5py
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer









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
app.secret_key = 'supersecretvalue123!'
serializer = URLSafeTimedSerializer(app.secret_key)



LOCK_PIN = 18  
GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK_PIN, GPIO.OUT)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK_PIN, GPIO.OUT)



def inspect_h5_model(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'layer_names' in f.attrs:
                print("Layer names:", [n.decode('utf8') for n in f.attrs['layer_names']])
            print("\nModel structure:")
            def print_structure(name, obj):
                print(name)
            f.visititems(print_structure)
    except Exception as e:
        print(f"Failed to inspect model: {e}")

print("Inspecting model file structure:")
inspect_h5_model('my_model.keras')
model = Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(1, activation='sigmoid')
])
try:
    model.load_weights('my_model.keras')
except Exception as e1:
    print(f"Failed to load .keras weights: {e1}")
    try:
        model = tf.keras.models.load_model('my_model.keras', compile=False)
    except Exception as e3:
        print(f"Failed to load complete model: {e3}")
        model = Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(1, activation='sigmoid')
        ])
        print("Using default model architecture without weights")



yolo_model = YOLO("yolov8n.pt")









#open camera
output_dir = "static/captured_images"
os.makedirs(output_dir, exist_ok=True)
camera = cv2.VideoCapture(0)
detecting = False
frame_lock = threading.Lock()
frame_to_show = None




def get_admin_email():
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT email FROM admin LIMIT 1")
        result = c.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"[ERROR] Could not fetch admin email: {e}")
        return None
    finally:
        conn.close()


def send_email_alert(image_path, alert_type="thief"):
    sender_email = ""
    receiver_email = get_admin_email()
    if not receiver_email:
        print("[ERROR] Admin email not found. Cannot send email.")
        return

    password = ""

    msg = MIMEMultipart()
    if alert_type == "suspicious_activity":
        msg['Subject'] = 'Suspicious Activity Detected Alert'
        body_text = f"A person with hidden face was detected. Image saved as {os.path.basename(image_path)}"
    else:
        msg['Subject'] = 'Thief Detected Alert'
        body_text = f"A potential thief was detected. Image saved as {os.path.basename(image_path)}"

    msg['From'] = sender_email
    msg['To'] = receiver_email

    body = MIMEText(body_text)
    msg.attach(body)

    with open(image_path, 'rb') as img_file:
        msg.attach(MIMEImage(img_file.read(), name=os.path.basename(image_path)))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            print("✅ Email sent to admin.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")







def send_reset_email(to_email, reset_link):
    sender_email = ""
    receiver_email = to_email
    password = ""  # Your Gmail App Password — keep safe!

    msg = MIMEMultipart()
    msg['Subject'] = 'Password Reset Instructions'
    msg['From'] = sender_email
    msg['To'] = receiver_email

    body_text = f"""\
Hello,

You (or someone else) requested a password reset.

Click the link below to reset your password:

{reset_link}

If you did not request this, please ignore this email.

Thanks,
Your Security Team
"""

    msg.attach(MIMEText(body_text, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            print("✅ Password reset email sent.")
    except Exception as e:
        print(f"❌ Failed to send password reset email: {e}")
        





BOT_TOKEN = ''
CHAT_ID = '7549315406'

def send_telegram_alert(message, image_path=None):
    # Send text message
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': CHAT_ID,
        'text': message
    }
    try:
        requests.post(url, data=payload)
        print("✅ Telegram alert sent.")
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")

    # Send image if provided
    if image_path:
        try:
            url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
            with open(image_path, 'rb') as photo:
                requests.post(url, data={'chat_id': CHAT_ID}, files={'photo': photo})
                print("✅ Telegram image sent.")
                
        except Exception as e:
            print(f"❌ Failed to send Telegram photo: {e}")




      




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








def log_intrusion(image_path, emotion, is_thief, is_suspicious, extra_info, name="Unknown", user_id=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute('''
        INSERT INTO intrusions (timestamp, image_path, emotion, is_thief, is_suspicious, detection_info, name, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, image_path, emotion, int(is_thief), int(is_suspicious), json.dumps(extra_info), name, user_id))

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
    if len(image_paths) < 5:
        print("❌ Please at least 5 image paths.")
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





def log_user_entry(user_id):
    import sqlite3
    from datetime import datetime

    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Must be called here again!
    c.execute("PRAGMA foreign_keys = ON")

    entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        c.execute('INSERT INTO user_entries (user_id, entry_time) VALUES (?, ?)', (user_id, entry_time))
        conn.commit()
        print(f"[LOGGED] Entry for user ID {user_id} at {entry_time}")
    except sqlite3.IntegrityError as e:
        print(f"[ERROR] Failed to log entry: {e}")
    finally:
        conn.close()





suspicious_start_time = None  # Global

def detect_suspicious_activity(frame, yolo_results, face_locations):
    global suspicious_start_time

    persons_detected = [box for box in yolo_results.boxes if int(box.cls[0]) == 0]
    loiter_threshold = 15  # seconds

    # Trigger if anyone has a hidden face (even 1 person)
    if len(persons_detected) > len(face_locations) and len(persons_detected) >= 1:
        if suspicious_start_time is None:
            suspicious_start_time = time.time()
        elif time.time() - suspicious_start_time >= loiter_threshold:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"{output_dir}/hidden_face_event_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)

            person_count = len(persons_detected)
            face_count = len(face_locations)
            event_type = "group" if person_count > 1 else "solo"

            print(f"[SECURITY] {event_type.capitalize()} hidden-face loitering detected — analyzing...")

            speak_warning("Security alert! You are being monitored. Please step back immediately.")

            is_thief = analyze_thief(image_path)

            log_intrusion(
                image_path=image_path,
                emotion="Unknown",
                is_thief=is_thief,
                is_suspicious=True,
                extra_info={
                    "trigger": f"hidden_face_{event_type}",
                    "persons": person_count,
                    "faces": face_count,
                    "loiter_duration": loiter_threshold
                }
            )

            if is_thief:
                send_email_alert(image_path, alert_type="thief_detected")
                send_telegram_alert(f"🚨 {event_type.capitalize()} with hidden face(s) detected!", image_path=image_path)
            else:
                print(f"[INFO] {event_type.capitalize()} flagged, no confirmed threat.")

            suspicious_start_time = None
            return True, persons_detected
        else:
            wait_time = int(time.time() - suspicious_start_time)
            print(f"[WAITING] Hidden-face loitering: {wait_time}s / {loiter_threshold}s")
    else:
        suspicious_start_time = None  # Reset if no more hidden faces

    return False, []




def speak_warning(message="Warning! Unauthorized access is prohibited. Please step back."):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()




def detect_faces():
    global frame_to_show, detecting, camera
    known_encodings, known_names = load_known_faces()
    image_count = 0
    last_save_time = 0
    save_interval = 10
    threshold = 0.3
    frame_count = 0

    analyzed_ids = {}
    group_entry_time = None
    group_loiter_threshold = 15  # seconds
    motion_threshold = 500000
    previous_frame = None

   
    if not hasattr(detect_faces, "unknown_start_time"):
        detect_faces.unknown_start_time = None

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Could not open camera")
        detecting = False
        return

    while detecting:
        try:
            frame_count += 1
            success, frame = camera.read()
            if not success:
                print("[ERROR] Failed to capture frame")
                time.sleep(0.1)
                continue

            original_frame = frame.copy()
            y1, y2 = 100, 400
            x1, x2 = 150, 500
            detection_frame = frame[y1:y2, x1:x2].copy()
            rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)

            # 🔍 Motion Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            motion_detected = True
            if previous_frame is not None:
                delta = cv2.absdiff(previous_frame, gray)
                thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                motion_score = cv2.countNonZero(thresh)
                motion_detected = motion_score > motion_threshold
            previous_frame = gray

            suspicious_activity_detected = False

            if frame_count % 3 == 0:
                try:
                    yolo_frame = cv2.resize(original_frame, (416, 416))
                    results = yolo_model.track(yolo_frame, persist=True, conf=0.25)

                    face_locations = face_recognition.face_locations(rgb_frame)
                    suspicious_activity_detected, _ = detect_suspicious_activity(
                        original_frame, results[0], face_locations
                    )

                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        track_id = int(box.id[0]) if box.id is not None else None

                        if cls == 0:
                            height_ratio = original_frame.shape[0] / 416
                            width_ratio = original_frame.shape[1] / 416
                            x1_box = int(box.xyxy[0][0].item() * width_ratio)
                            y1_box = int(box.xyxy[0][1].item() * height_ratio)
                            x2_box = int(box.xyxy[0][2].item() * width_ratio)
                            y2_box = int(box.xyxy[0][3].item() * height_ratio)

                            cv2.rectangle(original_frame, (x1_box, y1_box), (x2_box, y2_box), (255, 0, 255), 2)
                            cv2.putText(original_frame, f"ID: {track_id}", (x1_box, y1_box - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                            if track_id is not None:
                                if track_id in analyzed_ids and (time.time() - analyzed_ids[track_id]) < 15:
                                    continue
                                analyzed_ids[track_id] = time.time()

                except Exception as e:
                    print(f"[ERROR] YOLO tracking error: {e}")

            try:
                locations = face_recognition.face_locations(rgb_frame)
                encodings = face_recognition.face_encodings(rgb_frame, locations)
            except Exception as e:
                print(f"[ERROR] Face detection error: {e}")
                continue

            detected_names = []
            for (top, right, bottom, left), face_encoding in zip(locations, encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=threshold)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
                name = "Unknown"

                if best_match_index != -1 and matches[best_match_index]:
                    name = known_names[best_match_index]
                    detect_faces.unknown_start_time = None  
                    image_path = f"{output_dir}/{name}_{image_count}.jpg"
                    cv2.imwrite(image_path, detection_frame)
                    open_lock()
                    print(f"[INFO] Known person detected: {name}")

                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute("SELECT id FROM users WHERE name = ?", (name,))
                    result = c.fetchone()
                    conn.close()

                    if result:
                        user_id = result[0]
                        log_user_entry(user_id)
                else:
                    if detect_faces.unknown_start_time is None:
                        detect_faces.unknown_start_time = time.time()
                    elif time.time() - detect_faces.unknown_start_time >= 10:
                        image_count += 1
                        image_path = f"{output_dir}/Unknown_{image_count}.jpg"
                        print("[SECURITY] Unknown person loitering ≥ 20s — running analysis...")
                        try:
                            cv2.imwrite(image_path, detection_frame)
                            emotion_result = analyze_emotion(detection_frame)
                            thief = analyze_thief(image_path)
                            dominant_emotion = emotion_result[0]['dominant_emotion'] if emotion_result else "Unknown"

                            if thief:
                                send_email_alert(image_path)
                                send_telegram_alert("⚠️ Thief detected after loitering!", image_path=image_path)
                            log_intrusion(image_path, dominant_emotion, thief)
                        except Exception as e:
                            print(f"[ERROR] Failed analyzing loitering unknown person: {e}")
                        detect_faces.unknown_start_time = None  # Reset after analysis
                    else:
                        wait_time = int(time.time() - detect_faces.unknown_start_time)
                        print(f"[WAITING] Unknown person loitering: {wait_time}s / 20s")

                detected_names.append(name)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(detection_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(detection_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Final display
            display_frame = original_frame.copy()
            if not suspicious_activity_detected:
                display_frame[y1:y2, x1:x2] = detection_frame

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, f"Time: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if suspicious_activity_detected:
                cv2.putText(display_frame, "ALERT: Hidden Face Detected!", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            with frame_lock:
                frame_to_show = display_frame.copy()

        except Exception as e:
            print(f"[ERROR] Face detection loop error: {e}")
            time.sleep(0.1)

    camera.release()
    print("[INFO] Camera released after stopping detection.")





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
            is_suspicious INTEGER DEFAULT 0,
            detection_info TEXT,
            name TEXT DEFAULT 'Unknown',
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            entry_time TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()
initialize_db()
import os
print("Database path:", os.path.abspath('users.db'))















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






def validate_login(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM admin WHERE email=?", (email,))
    user = c.fetchone()
    conn.close()
    if user:
        stored_hashed_password = user[2]  
        if check_password_hash(stored_hashed_password, password):
            return user
    return None








@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        images = request.files.getlist("images")
        if len(images) < 5:
            return "Please upload at least 5 images."
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            image_paths = []
            for image in images:
                image_filename = secure_filename(image.filename)
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)
                image_paths.append(image_path)
            c.execute('''
                INSERT INTO users (name, email, phone)
                VALUES (?, ?, ?)
            ''', (name, email, phone))
            user_id = c.lastrowid
            for image_path in image_paths:
                image_loaded = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image_loaded)
                if not encodings:
                    conn.close()
                    return f"No face detected in {os.path.basename(image_path)}. Please upload clear face images."
                face_encoding = encodings[0]
                encoding_blob = face_encoding.tobytes()
                c.execute('''
                    INSERT INTO face_encodings (user_id, image_path, encoding)
                    VALUES (?, ?, ?)
                ''', (user_id, image_path, encoding_blob))
            conn.commit()
            conn.close()
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



@app.route('/data_report')
def data_report():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("PRAGMA foreign_keys = ON") 

    c.execute('''
        SELECT user_entries.id, user_entries.entry_time, users.name
        FROM user_entries
        JOIN users ON user_entries.user_id = users.id
        ORDER BY user_entries.entry_time DESC
        LIMIT 100
    ''')
    entries = c.fetchall()
    conn.close()
    return render_template('data_report.html', entries=entries)




@app.route('/delete_entry/<int:entry_id>', methods=['POST'])
def delete_entry(entry_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("DELETE FROM user_entries WHERE id = ?", (entry_id,))
        conn.commit()
        flash('Entry deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting entry: {e}', 'danger')
    finally:
        conn.close()
    return redirect(url_for('data_report'))




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




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = validate_login(email, password)
        if user:
            session['email'] = email
            return redirect(url_for('welcome'))
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html')





@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))






@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'email' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    email = session['email']
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    if request.method == 'POST':
        new_username = request.form['username'].strip()
        new_email = request.form['email'].strip()
        new_password = request.form['password'].strip()  
        confirm_password = request.form['confirm_password'].strip()  
        if new_password and new_password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('profile'))
        try:
            if new_password:
                hashed_password = generate_password_hash(new_password)
                c.execute("""
                    UPDATE admin SET username=?, email=?, password=? WHERE email=?
                """, (new_username, new_email, hashed_password, email))
            else:
                c.execute("""
                    UPDATE admin SET username=?, email=? WHERE email=?
                """, (new_username, new_email, email))
            conn.commit()
            session['email'] = new_email  
            flash('Profile updated successfully!', 'success')
        except sqlite3.Error as e:
            flash(f'Error updating profile: {e}', 'danger')
    c.execute("SELECT * FROM admin WHERE email=?", (session['email'],))
    admin = c.fetchone()
    conn.close()

    return render_template('profile.html', admin=admin)







@app.route('/welcome')
def welcome():
    return render_template('welcome.html')







@app.route("/dashboard")
def dashboard():
    temp, hum = fetch_weather_data(CITY)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("PRAGMA foreign_keys = ON")
    
    c.execute('''
        SELECT user_entries.entry_time, users.name
        FROM user_entries
        JOIN users ON user_entries.user_id = users.id
        ORDER BY user_entries.entry_time DESC
        LIMIT 1
    ''')
    last_entry = c.fetchone()
    conn.close()
    if last_entry:
        entry_time_str = datetime.strptime(last_entry[0], '%Y-%m-%d %H:%M:%S').strftime('%b %d, %H:%M')
        last_entry_display = {
            "time": entry_time_str,
            "user": last_entry[1]
        }
    else:
        last_entry_display = {
            "time": "No entries",
            "user": ""
        }

    data = {
        "temperature": temp or 0,
        "humidity": hum or 0,
        "fire_status": "No Fire",
        "last_entry": last_entry_display,
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










@app.route('/intrusions', methods=['GET', 'POST'])
def view_intrusions():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Handle deletion if POST
    if request.method == 'POST':
        delete_id = request.form.get('delete_id')
        if delete_id:
            c.execute("SELECT image_path FROM intrusions WHERE id = ?", (delete_id,))
            image = c.fetchone()
            if image and image[0]:
                image_path = os.path.join('static', image[0])
                if os.path.exists(image_path):
                    os.remove(image_path)

            c.execute("DELETE FROM intrusions WHERE id = ?", (delete_id,))
            conn.commit()
            flash('Intrusion log deleted successfully.')
            conn.close()
            return redirect(url_for('view_intrusions'))

    # Filters
    name = request.args.get('name', '').strip()
    is_thief = request.args.get('is_thief', '')

    query = "SELECT id, timestamp, emotion, is_thief, name, image_path FROM intrusions WHERE 1=1"
    params = []

    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if is_thief in ('0', '1'):
        query += " AND is_thief = ?"
        params.append(is_thief)

    query += " ORDER BY timestamp DESC"
    c.execute(query, params)
    logs = c.fetchall()
    conn.close()

    return render_template('intrusions.html', logs=logs)



@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password_page():
    if request.method == 'POST':
        email = request.form['email']

        # Connect to database directly
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row  # optional: for dict-like rows
        c = conn.cursor()

        # Query for the admin user
        c.execute("SELECT * FROM admin WHERE email=?", (email,))
        user = c.fetchone()

        if user:
            token = serializer.dumps(email, salt='password-reset-salt')
            reset_link = url_for('reset_password', token=token, _external=True)
            print(f"Password reset link for {email}: {reset_link}")
            send_reset_email(email, reset_link)
        else:
           
            flash('If this email is registered, you will receive instructions.', 'info')

        conn.close()  
        return redirect(url_for('forgot_password_page'))

    return render_template('forgot_password.html')




@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except (SignatureExpired, BadSignature):
        flash('The reset link is invalid or expired.', 'danger')
        return redirect(url_for('forgot_password_page'))  # Adjust route name if needed

    if request.method == 'POST':
        password = request.form['password']
        password_confirm = request.form['password_confirm']

        if password != password_confirm:
            flash('Passwords do not match.', 'warning')
            return redirect(request.url)

        new_hash = generate_password_hash(password)

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("UPDATE admin SET password=? WHERE email=?", (new_hash, email))
        conn.commit()
        conn.close()

        flash('Your password has been reset. Please login.', 'success')
        return redirect(url_for('login'))
  

    return render_template('reset_password.html')





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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
