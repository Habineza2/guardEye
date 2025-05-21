
from flask import Flask, flash, logging, redirect, render_template, Response, request, session, url_for
import cv2
from werkzeug.security import generate_password_hash, check_password_hash
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
app.secret_key = ''






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







def send_email_alert(image_path, alert_type="thief"):
    sender_email = ""
    receiver_email = "your_email@example.com"
    password = ""
    msg = MIMEMultipart()
    
    if alert_type == "suspicious_activity":
        msg['subject'] = 'Suspicious Activity Detected Alert'
        body_text = f"A person with hidden face was detected. Image saved as {os.path.basename(image_path)}"
    else:
        msg['subject'] = 'Thief Detected Alert'
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








def log_intrusion(image_path, emotion="Unknown", is_thief=False, name="Unknown", is_suspicious=False, extra_info=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Check if the detection_info column exists, add it if it doesn't
    c.execute("PRAGMA table_info(intrusions)")
    columns = [info[1] for info in c.fetchall()]
    
    if 'detection_info' not in columns:
        c.execute("ALTER TABLE intrusions ADD COLUMN detection_info TEXT")
    
    if 'is_suspicious' not in columns:
        c.execute("ALTER TABLE intrusions ADD COLUMN is_suspicious INTEGER DEFAULT 0")
    
    # Convert extra_info dict to string for storage
    detection_info = str(extra_info) if extra_info else None
    
    c.execute("""
        INSERT INTO intrusions 
        (timestamp, image_path, emotion, is_thief, name, is_suspicious, detection_info) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        image_path, 
        emotion, 
        int(is_thief), 
        name, 
        int(is_suspicious), 
        detection_info
    ))
    
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








def detect_suspicious_activity(frame, yolo_results, face_locations):
    """
    Detect suspicious activity by comparing the number of persons detected by YOLO
    with the number of faces detected by face_recognition.
    """
    # Extract person detections from YOLO results (class 0 is person in COCO dataset)
    persons_detected = [box for box in yolo_results.boxes if int(box.cls[0]) == 0]
    
    # If more persons than faces detected, log as suspicious activity
    if len(persons_detected) > len(face_locations):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"{output_dir}/suspicious_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        
        # Log the suspicious activity
        extra_info = {
            'persons_detected': len(persons_detected),
            'faces_detected': len(face_locations),
            'confidence_scores': [float(box.conf[0]) for box in persons_detected]
        }
        
        log_intrusion(
            image_path,
            "Unknown",
            is_thief=False,
            is_suspicious=True,
            extra_info=extra_info
        )
        
        # Send alert
        send_email_alert(image_path, alert_type="suspicious_activity")
        
        print(f"[ALERT] Suspicious activity detected - {len(persons_detected)} persons but only {len(face_locations)} faces visible")
        return True, persons_detected
    
    return False, []




def detect_faces():
    global frame_to_show, detecting
    
    # Load known face encodings and names from the database
    known_encodings, known_names = load_known_faces()
    
    image_count = 0
    last_save_time = 0
    save_interval = 10 
    threshold = 0.2
    frame_count = 0
    
    while detecting:
        try:
            frame_count += 1
            success, frame = camera.read()
            if not success:
                print("[ERROR] Failed to capture frame from camera")
                time.sleep(0.1)
                continue
                
            # Store the original frame for YOLO detection
            original_frame = frame.copy()
            
            # Crop frame for face detection as in original code
            y1, y2 = 100, 400  
            x1, x2 = 150, 500  
            detection_frame = frame[y1:y2, x1:x2].copy()
            
            # Convert for face recognition
            rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection every 3rd frame to maintain performance
            suspicious_activity_detected = False
            suspicious_boxes = []
            if frame_count % 3 == 0:
                try:
                    # Resize frame for YOLO processing to improve performance
                    yolo_frame = cv2.resize(original_frame, (416, 416))
                    
                    # Run YOLO detection with confidence threshold of 0.25
                    results = yolo_model(yolo_frame, conf=0.25)
                    
                    # Get face locations for comparison with person detections
                    face_locations = face_recognition.face_locations(rgb_frame)
                    
                    # Check for suspicious activity (persons without visible faces)
                    suspicious_activity_detected, suspicious_boxes = detect_suspicious_activity(
                        original_frame, results[0], face_locations
                    )
                    
                    # If suspicious activity detected, draw boxes on the frame for display
                    if suspicious_activity_detected and len(suspicious_boxes) > 0:
                        # Scale boxes back to original frame size
                        height_ratio = original_frame.shape[0] / 416
                        width_ratio = original_frame.shape[1] / 416
                        
                        for box in suspicious_boxes:
                            # Extract coordinates
                            x1_box = int(box.xyxy[0][0].item() * width_ratio)
                            y1_box = int(box.xyxy[0][1].item() * height_ratio)
                            x2_box = int(box.xyxy[0][2].item() * width_ratio)
                            y2_box = int(box.xyxy[0][3].item() * height_ratio)
                            
                            # Draw red box for suspicious activity
                            cv2.rectangle(original_frame, (x1_box, y1_box), (x2_box, y2_box), (0, 0, 255), 2)
                            cv2.putText(original_frame, "Hidden Face", (x1_box, y1_box - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"[ERROR] YOLO detection error: {e}")
                    # Continue with face recognition even if YOLO fails
            
            try:
                locations = face_recognition.face_locations(rgb_frame)
                encodings = face_recognition.face_encodings(rgb_frame, locations)
            except Exception as e:
                print(f"[ERROR] Face detection error: {e}")
                continue
            for (top, right, bottom, left), face_encoding in zip(locations, encodings):
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
                    
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute("SELECT id FROM users WHERE name = ?", (name,))
                    result = c.fetchone()
                    conn.close()

                    if result:
                      user_id = result[0]
                      log_user_entry(user_id)
                    else:
                      print(f"[WARN] No user_id found for {name}")

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

                # Draw rectangle and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(detection_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(detection_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
            # Combine the detection frame back into original frame for display
            display_frame = original_frame.copy()
            if suspicious_activity_detected:
                # We already have the YOLO boxes drawn on original_frame
                pass
            else:
                # Insert detection_frame back into the display frame
                display_frame[y1:y2, x1:x2] = detection_frame
                
            # Add timestamp and status message to the frame
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, f"Time: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if suspicious_activity_detected:
                cv2.putText(display_frame, "ALERT: Hidden Face Detected!", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Update the frame to show
            with frame_lock:
                frame_to_show = display_frame.copy()
                
        except Exception as e:
            print(f"[ERROR] Error in face detection loop: {e}")
            time.sleep(0.1)






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
        new_password = request.form['password'].strip()  # new password entered by admin (optional)
        confirm_password = request.form['confirm_password'].strip()  # confirm new password

        # Validation
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
            session['email'] = new_email  # update session email if changed
            flash('Profile updated successfully!', 'success')
        except sqlite3.Error as e:
            flash(f'Error updating profile: {e}', 'danger')

    # Always fetch latest data
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
    
    # Connect to your database and fetch the latest user entry
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
    last_entry = c.fetchone()  # Will be (entry_time, name) or None
    
    conn.close()
    
    if last_entry:
        # Format the datetime string nicely: e.g. May 15, 18:10
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






@app.route('/intrusions')
def view_intrusions():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Check if is_suspicious column exists
    c.execute("PRAGMA table_info(intrusions)")
    columns = [info[1] for info in c.fetchall()]
    
    if 'is_suspicious' in columns:
        c.execute("""
            SELECT id, timestamp, image_path, emotion, is_thief, is_suspicious, detection_info, name
            FROM intrusions 
            ORDER BY timestamp DESC
        """)
    else:
        c.execute("""
            SELECT id, timestamp, image_path, emotion, is_thief, name
            FROM intrusions 
            ORDER BY timestamp DESC
        """)
    
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


