import csv
import os
import io
import pandas as pd
import cv2
import json
import time
import pickle
import sqlite3
import threading
from fpdf import FPDF
import requests
import numpy as np
import smtplib
import logging
import pyttsx3
from PIL import Image
from datetime import datetime
from flask import Flask, flash, jsonify, redirect, render_template, Response, request, send_from_directory, session, url_for, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import face_recognition
from tflite_runtime.interpreter import Interpreter
import tflite_runtime.interpreter as tflite
from RPLCD.i2c import CharLCD



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
app.secret_key = 'emmy'
serializer = URLSafeTimedSerializer(app.secret_key)




tflite_model_path = "my_model.tflite"
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded.")
print("Input details:", input_details)
print("Output details:", output_details)




LOCK_PIN = 18  
GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK_PIN, GPIO.OUT)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK_PIN, GPIO.OUT)

fire_status = "Safe"

current_frame = None
frame_lock = threading.Lock()





#open camera
output_dir = "static/captured_images"
os.makedirs(output_dir, exist_ok=True)
#camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

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
CHAT_ID = ''

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


def analyze_thief(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unreadable")

        img = cv2.resize(img, (150, 150))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        prediction = output[0][0]
        is_thief = prediction >= 0.5

        print(f"[Thief Detection] Prediction: {prediction:.2f} - {'Thief' if is_thief else 'Not Thief'}")
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



def log_intrusion(image_path, is_thief, is_suspicious=0):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        c.execute('''
            INSERT INTO intrusions (timestamp, image_path, is_thief, is_suspicious)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, image_path, int(is_thief), int(is_suspicious)))
        conn.commit()
        print(f"[LOGGED] Intrusion at {timestamp}, thief: {is_thief}, suspicious: {is_suspicious}")
    except Exception as e:
        print(f"[ERROR] Failed to log intrusion: {e}")
    finally:
        conn.close()








def open_lock():
    initialize_gpio()  
    GPIO.output(LOCK_PIN, GPIO.LOW)  
    print("Door unlocked.")
    time.sleep(10)  
    GPIO.output(LOCK_PIN, GPIO.HIGH)  
    print("Door locked.")



gpio_initialized = False

def initialize_gpio():
    global gpio_initialized
    if gpio_initialized:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LOCK_PIN, GPIO.OUT)
    GPIO.output(LOCK_PIN, GPIO.HIGH)  # Ensure locked state by default
    gpio_initialized = True





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
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
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

        




# Load label map (assuming labelmap.txt is in same folder)
def load_labels(path='labelmap.txt'):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels('labelmap.txt')




def detect_suspicious_activity(frame, detected_persons, face_locations, recent_alerts):
    if len(detected_persons) > len(face_locations):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(output_dir, f"suspicious_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)

        extra_info = {
            'persons_detected': len(detected_persons),
            'faces_detected': len(face_locations),
            'confidence_scores': [p['confidence'] for p in detected_persons]
        }

        thief_detected = False

        for i, person in enumerate(detected_persons):
            try:
                x1, y1, x2, y2 = person['box']
                crop_key = f"{x1}_{y1}_{x2}_{y2}"
                now = time.time()

                if crop_key in recent_alerts and now - recent_alerts[crop_key] < 10:
                    continue

                cropped = frame[y1:y2, x1:x2]
                cropped_path = os.path.join(output_dir, f"person_{timestamp}_{i}.jpg")

                if cropped.size == 0:
                    print(f"[WARNING] Empty crop at {x1}, {y1}, {x2}, {y2}, skipping...")
                    continue

                if not cv2.imwrite(cropped_path, cropped):
                    print(f"[ERROR] Failed to save image at {cropped_path}")
                    continue

                if analyze_thief(cropped_path):
                    thief_detected = True
                    print(f"[ALERT] Thief predicted in person {i}")

                    send_email_alert(image_path, alert_type="thief_detected")
                    send_telegram_alert("🚨 Thief detected among suspicious persons!", image_path=image_path)
                else:
                    print(f"[INFO] Person {i} suspicious but no thief detected.")

                recent_alerts[crop_key] = now

            except Exception as e:
                print(f"[ERROR] Failed to process person {i}: {e}")
                continue

        # log_intrusion(
        #     image_path,
        #     "Unknown",
        #     is_thief=thief_detected,
        #     is_suspicious=True,
        #     extra_info=extra_info
        # )
        log_intrusion(image_path, is_thief=thief_detected, is_suspicious=1)


        return True, detected_persons

    return False, []







frame_to_show = None
frame_lock = threading.Lock()
detecting = False
# output_dir = "output"
yolo_model = lambda x, conf=0.25: [[]]  
camera = None





FLAME_SENSOR_PIN = 17
flame_detecting = False
flame_thread = None
def log_fire_event():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute('''
        INSERT INTO fire_logs (timestamp, detection_info)
        VALUES (?, ?)
    ''', (timestamp, json.dumps({'alert': 'Fire detected'})))

    conn.commit()
    conn.close()
    print(f"[LOG] 🔥 Fire event logged to fire_logs at {timestamp}")






BUZZER_PIN = 23   

last_fire_time = 0
def monitor_flame_sensor():
    global flame_detecting, fire_status, last_fire_time
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(FLAME_SENSOR_PIN, GPIO.IN)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    lcd = CharLCD('PCF8574', 0x27)

    print("[INFO] Flame sensor monitoring started.")

    while flame_detecting:
        if GPIO.input(FLAME_SENSOR_PIN) == 0:
            print("🔥🔥🔥 Fire detected!")
            fire_status = "🔥 Fire Detected!"
            last_fire_time = time.time()

            lcd.clear()
            lcd.write_string("🔥 Fire Detected!")

            log_fire_event()
            send_telegram_alert("🚨 FIRE DETECTED at location! Immediate attention needed.")

            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(3)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            lcd.clear()

        # Keep "Fire Detected" status for 5 seconds after last detection
        elif time.time() - last_fire_time > 5:
            fire_status = "Safe"

        time.sleep(0.5)

    print("[INFO] Flame sensor monitoring stopped.")
    GPIO.cleanup()



def start_flame_monitor():
    global flame_detecting, flame_thread
    if not flame_detecting:
        flame_detecting = True
        flame_thread = threading.Thread(target=monitor_flame_sensor, daemon=True)
        flame_thread.start()
        print("[INFO] Flame sensor monitoring thread started.")


start_flame_monitor()







def capture_frames():
    global current_frame, detecting, camera

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("[ERROR] Could not open camera")
        detecting = False
        return

    while detecting:
        ret, frame = camera.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
        time.sleep(0.015)

    camera.release()
    print("[INFO] Camera released")




def get_user_id_by_name(name):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE name = ?", (name,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None


def get_rfid_uid(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT rfid_uid FROM users WHERE id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None


def verify_rfid(uid_expected, timeout=10):
    from mfrc522 import SimpleMFRC522
    import time

    reader = SimpleMFRC522()
    print(f"👉 Please scan your RFID card within {timeout} seconds...")

    start_time = time.time()
    scanned_uid = None

    try:
        while time.time() - start_time < timeout:
            id = reader.read_id()
            if id:
                scanned_uid = str(id)
                print(f"✅ Scanned UID: {scanned_uid}")
                break
            time.sleep(0.5)

    except Exception as e:
        print(f"[ERROR] RFID scan failed: {e}")

    # finally:
    #     GPIO.cleanup()

    if not scanned_uid:
        print("[WARNING] RFID scan timed out.")
        return False

    return scanned_uid == str(uid_expected)




def detect_faces():
    global current_frame, frame_to_show, detecting

    known_encodings, known_names = load_known_faces()
    image_count = 0
    last_save_time = 0
    save_interval = 10  # seconds
    threshold = 0.33
    frame_count = 0
    detection_delay_seconds = 5
    start_time = time.time()
    recent_alerts = {}
    lcd = CharLCD('PCF8574', 0x27)

    lock_open = False
    last_unlock_time = 0
    lock_cooldown = 5  # seconds

    last_suspicious_detected_time = 0
    suspicious_cooldown = 10  # seconds

    print("[INFO] Detection thread running")

    # Initialize TFLite SSD MobileNet interpreter
    interpreter = tflite.Interpreter(model_path="detect.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]

    print(f"[INFO] TFLite input shape: {input_shape}")

    while detecting:
        with frame_lock:
            if current_frame is None:
                continue
            full_frame = current_frame.copy()

        frame_count += 1
        display_frame = full_frame.copy()

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

        current_time = time.time()

        if face_locations:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=threshold)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if face_distances.size > 0 else -1

                name = "Unknown"
                if best_match_index != -1 and matches[best_match_index]:
                    name = known_names[best_match_index]
                    user_id = get_user_id_by_name(name)  # You can make a helper to fetch user_id from users table
                    expected_uid = get_rfid_uid(user_id)  # fetch assigned RFID UID

                    if expected_uid:
                        lcd.clear()
                        lcd.write_string("Scan your card")
                        if verify_rfid(expected_uid):
                            lcd.clear()
                            lcd.write_string("Access granted")
                            open_lock()
                            print(f"[INFO] 2FA passed. Door unlocked for {name}.")
                            log_user_entry(user_id) 
                        else:
                            print(f"[ALERT] RFID verification failed for {name}!")
                            lcd.clear()
                            lcd.write_string("Access denied")
                    else:
                        print(f"[WARNING] No RFID UID registered for {name}.")
                        lcd.clear()
                        lcd.write_string("Card not registered")
                    time.sleep(3)
                    lcd.clear()
    
                else:
                    if current_time - last_save_time > save_interval:
                        image_count += 1
                        image_path = os.path.join(output_dir, f"Unknown_{image_count}.jpg")
                        print("[INFO] Unknown person detected")

                        try:
                            cv2.imwrite(image_path, detection_frame)
                            # emotion_result = analyze_emotion(detection_frame)
                            thief = analyze_thief(image_path)
                            # dominant_emotion = emotion_result[0]['dominant_emotion'] if emotion_result else "Unknown"

                            if thief:
                                send_email_alert(image_path)
                                send_telegram_alert("⚠️ Thief detected!", image_path)
                                print("[ALERT] Thief detected!")

                            log_intrusion(image_path, is_thief=thief)

                        except Exception as e:
                            print(f"[ERROR] Error processing unknown face: {e}")

                        last_save_time = current_time

                # Draw rectangle on display frame
                top += y1
                bottom += y1
                left += x1
                right += x1

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(display_frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        else:
            if current_time - start_time > detection_delay_seconds and frame_count % 3 == 0:
                print("[WARNING] No face detected. Running TFLite SSD...")

                if current_time - last_suspicious_detected_time > suspicious_cooldown:
                    try:
                        # Prepare input for TFLite
                        img_resized = cv2.resize(full_frame, (width, height))
                        input_data = np.expand_dims(img_resized, axis=0)

                        if input_details[0]['dtype'] == np.float32:
                            input_data = (np.float32(input_data) - 127.5) / 127.5

                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()

                        # Get detections
                        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
                        classes = interpreter.get_tensor(output_details[1]['index'])[0]
                        scores = interpreter.get_tensor(output_details[2]['index'])[0]
                        num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

                        im_h, im_w, _ = full_frame.shape
                        max_detections = min(num_detections, len(scores))

                        detected = False
                        for i in range(max_detections):
                            if scores[i] > 0.4 and int(classes[i]) == 0:
                                ymin, xmin, ymax, xmax = boxes[i]
                                x1_, y1_, x2_, y2_ = int(xmin * im_w), int(ymin * im_h), int(xmax * im_w), int(ymax * im_h)

                                # Draw rectangle
                                cv2.rectangle(display_frame, (x1_, y1_), (x2_, y2_), (0, 0, 255), 2)
                                cv2.putText(display_frame, "Suspicious", (x1_, y1_ - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                                # Crop and save detected person
                                person_crop = full_frame[y1_:y2_, x1_:x2_]
                                image_path = f"{output_dir}/Suspicious_{image_count}.jpg"
                                cv2.imwrite(image_path, person_crop)

                                thief = analyze_thief(image_path)
                                if thief:
                                    send_email_alert(image_path)
                                    send_telegram_alert("⚠️ Thief detected via TFLite SSD!", image_path)
                                    # log_intrusion(image_path, "Face Hidden", thief)
                                    log_intrusion(image_path, is_thief=thief, is_suspicious=1)

                                    print("[ALERT] Thief detected via TFLite SSD!")
                                else:
                                    print("[INFO] Suspicious person detected.")

                                image_count += 1
                                detected = True

                        if detected:
                            last_suspicious_detected_time = current_time
                        else:
                            print("[INFO] No persons detected by TFLite SSD.")

                    except Exception as e:
                        print(f"[ERROR] TFLite SSD detection error: {e}")

        with frame_lock:
            frame_to_show = display_frame.copy()

        time.sleep(0.03)

    print("[INFO] Detection thread stopped")



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
            is_thief INTEGER,
            is_suspicious INTEGER DEFAULT 0
           
           
            
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
            flash("Please upload at least 5 images.", "error")
            return redirect(url_for('register'))

        # Save images first
        image_paths = []
        for image in images:
            image_filename = secure_filename(image.filename)
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path)
            image_paths.append(image_path)

        # After images saved, flash a message to prompt RFID scan
        flash("Images uploaded successfully! Please scan your RFID card now.", "info")

        # Render the register page but now waiting for RFID scan
        # You can pass image paths or store in session if needed

        # For demonstration, try scanning RFID now (blocking call)
        from mfrc522 import SimpleMFRC522
        reader = SimpleMFRC522()

        try:
            rfid_uid = str(reader.read_id())
            flash(f"RFID card detected with UID: {rfid_uid}", "success")
        except Exception as e:
            flash(f"Failed to read RFID card: {e}", "error")
            GPIO.cleanup()
            return redirect(url_for('register'))

        GPIO.cleanup()

        # Now continue with saving user and encodings as you had

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()

            c.execute('''
                INSERT INTO users (name, email, phone, rfid_uid)
                VALUES (?, ?, ?, ?)
            ''', (name, email, phone, rfid_uid))
            user_id = c.lastrowid

            for image_path in image_paths:
                image_loaded = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image_loaded)
                if not encodings:
                    conn.close()
                    flash(f"No face detected in {os.path.basename(image_path)}. Please upload clear face images.", "error")
                    return redirect(url_for('register'))

                face_encoding = encodings[0]
                encoding_blob = face_encoding.tobytes()

                c.execute('''
                    INSERT INTO face_encodings (user_id, image_path, encoding)
                    VALUES (?, ?, ?)
                ''', (user_id, image_path, encoding_blob))

            conn.commit()
            conn.close()

            # Redirect to dashboard or show success message
            flash("User registered successfully!", "success")
            # return redirect(url_for('dashboard'))

        except Exception as e:
            flash(f"Registration failed: {e}", "error")
            return redirect(url_for('register'))

    # GET request
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



def get_intrusion_stats():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Get total number of intrusions
    cursor.execute("SELECT COUNT(*) FROM intrusions")
    total_intrusions = cursor.fetchone()[0]

    # Get latest intrusion timestamp
    cursor.execute("SELECT timestamp FROM intrusions ORDER BY timestamp DESC LIMIT 1")
    result = cursor.fetchone()
    last_intrusion_time = (
        datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S').strftime('%b %d, %H:%M')
        if result else "No intrusions"
    )

    conn.close()
    return total_intrusions, last_intrusion_time




@app.route("/dashboard")
def dashboard():
    total_intrusions, last_intrusion_time = get_intrusion_stats()
   
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
        
        "fire_status": "No Fire",
        "last_entry": last_entry_display,
        "last_intrusion_time": last_intrusion_time,
        "total_intrusions": total_intrusions,
        
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

    if request.method == 'POST':
        delete_id = request.form.get('delete_id')
        if delete_id:
            c.execute("SELECT image_path FROM intrusions WHERE id = ?", (delete_id,))
            image = c.fetchone()
            if image and image[0] and os.path.exists(image[0]):
                os.remove(image[0])
            c.execute("DELETE FROM intrusions WHERE id = ?", (delete_id,))
            conn.commit()
            flash('Intrusion log deleted.')
            conn.close()
            return redirect(url_for('view_intrusions'))

    # Filters
    is_thief = request.args.get('is_thief', '')
    from_date = request.args.get('from_date', '')
    to_date = request.args.get('to_date', '')

    query = "SELECT id, timestamp, image_path, is_thief, is_suspicious FROM intrusions WHERE 1=1"
    params = []

    if is_thief in ('0', '1'):
        query += " AND is_thief = ?"
        params.append(is_thief)

    if from_date:
        query += " AND date(timestamp) >= date(?)"
        params.append(from_date)
    if to_date:
        query += " AND date(timestamp) <= date(?)"
        params.append(to_date)

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



















def gen_frames():
    global frame_to_show
    while True:
        with frame_lock:
            if frame_to_show is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame_to_show)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




capture_thread = None
detection_thread = None

def start_detection():
    global detecting, capture_thread, detection_thread

    if not detecting:
        initialize_gpio()  # <-- added line: make sure GPIO is ready before threads

        detecting = True
        print("[INFO] Starting capture and detection threads...")

        capture_thread = threading.Thread(target=capture_frames, daemon=True)
        detection_thread = threading.Thread(target=detect_faces, daemon=True)

        capture_thread.start()
        detection_thread.start()
    else:
        print("[INFO] Detection is already running.")


def stop_detection():
    global detecting

    if detecting:
        detecting = False
        print("[INFO] Stopping detection...")

        # Wait for threads to finish gracefully
        if capture_thread is not None:
            capture_thread.join(timeout=1)
        if detection_thread is not None:
            detection_thread.join(timeout=1)

        print("[INFO] Detection stopped.")
    else:
        print("[INFO] Detection is not running.")


@app.route('/start')
def start_route():
    start_detection()
    return "Detection started."

@app.route('/stop')
def stop_route():
    stop_detection()
    return "Detection stopped."


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/export_intrusions/<filetype>')
def export_intrusions(filetype):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        SELECT id, timestamp, is_thief, is_suspicious, image_path FROM intrusions ORDER BY timestamp DESC
    ''')
    logs = c.fetchall()
    conn.close()

    # CSV Export
    if filetype == 'csv':
        filepath = 'intrusions_export.csv'
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Timestamp', 'Is Thief', 'Is Suspicious', 'Image Path'])
            writer.writerows(logs)
        return send_file(filepath, as_attachment=True)

    # Excel Export
    elif filetype == 'excel':
        df = pd.DataFrame(logs, columns=['ID', 'Timestamp', 'Is Thief', 'Is Suspicious', 'Image Path'])
        filepath = 'intrusions_export.xlsx'
        df.to_excel(filepath, index=False)
        return send_file(filepath, as_attachment=True)

    # PDF Export
    elif filetype == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Intrusion Logs", ln=True, align='C')
        pdf.ln(10)

        for log in logs:
            text = (
                f"ID: {log[0]} | "
                f"Time: {log[1]} | "
                f"Thief: {'Yes' if log[2] else 'No'} | "
                f"Suspicious: {'Yes' if log[3] else 'No'} | "
                f"Image: {log[4]}"
            )
            pdf.multi_cell(0, 10, txt=text)
            pdf.ln(1)

        filepath = "intrusions_export.pdf"
        pdf.output(filepath)
        return send_file(filepath, as_attachment=True)

    else:
        return "Unsupported format", 400




@app.route('/export_users/<filetype>')
def export_users(filetype):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, name, email, phone FROM users ORDER BY id ASC")
    users = c.fetchall()
    conn.close()

    if filetype == 'csv':
        filepath = 'users_export.csv'
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Name', 'Email', 'Phone'])
            writer.writerows(users)
        return send_file(filepath, as_attachment=True)

    elif filetype == 'excel':
        df = pd.DataFrame(users, columns=['ID', 'Name', 'Email', 'Phone'])
        filepath = 'users_export.xlsx'
        df.to_excel(filepath, index=False)
        return send_file(filepath, as_attachment=True)

    elif filetype == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Registered Users", ln=True, align='C')
        pdf.ln(10)
        for user in users:
            line = f"ID: {user[0]} | Name: {user[1]} | Email: {user[2]} | Phone: {user[3]}"
            pdf.multi_cell(0, 10, txt=line)
        filepath = 'users_export.pdf'
        pdf.output(filepath)
        return send_file(filepath, as_attachment=True)

    else:
        return "Unsupported export format", 400
    
    
    
    
    
@app.route('/export_user_entries/<filetype>')
def export_user_entries(filetype):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        SELECT user_entries.id, user_entries.entry_time, users.name
        FROM user_entries
        JOIN users ON user_entries.user_id = users.id
        ORDER BY user_entries.entry_time DESC
    ''')
    entries = c.fetchall()
    conn.close()

    if filetype == 'csv':
        filepath = 'user_entries_export.csv'
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Entry ID', 'Time', 'User Name'])
            writer.writerows(entries)
        return send_file(filepath, as_attachment=True)

    elif filetype == 'excel':
        df = pd.DataFrame(entries, columns=['Entry ID', 'Time', 'User Name'])
        filepath = 'user_entries_export.xlsx'
        df.to_excel(filepath, index=False)
        return send_file(filepath, as_attachment=True)

    elif filetype == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="User Entry Logs", ln=True, align='C')
        pdf.ln(10)
        for entry in entries:
            line = f"ID: {entry[0]} | Time: {entry[1]} | Name: {entry[2]}"
            pdf.multi_cell(0, 10, txt=line)
        filepath = 'user_entries_export.pdf'
        pdf.output(filepath)
        return send_file(filepath, as_attachment=True)

    else:
        return "Unsupported export format", 400



@app.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory('/home/pi/Undergraduate projects/image', filename)


def get_intrusions_per_hour():
    conn = sqlite3.connect('users.db')
    cursor = conn.execute("SELECT timestamp FROM intrusions")
    timestamps = [row[0] for row in cursor.fetchall()]
    conn.close()

    counts = {}
    for ts_str in timestamps:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        hour_key = dt.replace(minute=0, second=0, microsecond=0)
        counts[hour_key] = counts.get(hour_key, 0) + 1

    sorted_hours = sorted(counts.keys())
    labels = [h.strftime("%Y-%m-%d %H:%M") for h in sorted_hours]
    values = [counts[h] for h in sorted_hours]

    return labels, values  


@app.route('/data')
def data():
    labels, values = get_intrusions_per_hour()
    return jsonify({'labels': labels, 'values': values,   'fire_status': fire_status})

if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)


