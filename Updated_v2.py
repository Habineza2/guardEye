import tkinter as tk
from tkinter import messagebox, font
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np
import os
import face_recognition
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from deepface import DeepFace
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import pyttsx3


def send_email_alert(image_path):
    sender_email = ""
    receiver_email = "your_email@example.com"  
    password = ""
    msg = MIMEMultipart()
    msg['subject'] = 'Thief Detected Alert'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    body = MIMEText(f"A potential thief was detected. Image saved as {os.path.basename(image_path)}")
    msg.attach(body)
    with open(image_path, 'rb') as img_file:
        image_data = img_file.read()
    image = MIMEImage(image_data, name=os.path.basename(image_path))
    msg.attach(image)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            print("Email sent.")
    except Exception as e:
        print(f"Failed to send email: {e}")


model = load_model("my_model.h5")


def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []
    for filename in os.listdir(known_faces_dir):
        file_path = os.path.join(known_faces_dir, filename)
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = face_recognition.load_image_file(file_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(filename.split('.')[0])
    return known_encodings, known_names


def analyze_emotion(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_filename = "temp_frame.jpg"
        cv2.imwrite(temp_filename, frame)
        analysis = DeepFace.analyze(temp_filename, actions=['emotion'], enforce_detection=False)
        return analysis
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return None


def analyze_thief(image_path):
    img_width, img_height = 150, 150
    threshold = 0.5
    image = cv2.imread(image_path)
    img = cv2.resize(image, (img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    try:
        prediction = model.predict(img)
        if prediction >= threshold:
            print(f"Thief detected in {image_path}")
            return True
        else:
            print(f"No thief detected in {image_path}")
            return False
    except Exception as e:
        print(f"Error analyzing thief: {e}")
        return False


def speak_warning():
    engine = pyttsx3.init()
    engine.say("Warning! You are too close. Please step back or I shut.")
    engine.runAndWait()


class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        self.title_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.button_font = font.Font(family="Helvetica", size=12)
        self.title_label = tk.Label(root, text="Face Detection System", font=self.title_font, bg="#f0f0f0")
        self.title_label.pack(pady=10)
        self.button_frame = tk.Frame(root, bg="#f0f0f0")
        self.button_frame.pack(pady=10)

        self.start_button = tk.Button(self.button_frame, text="Start Detection", command=self.start_detection, font=self.button_font, bg="#4CAF50", fg="white")
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(self.button_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED, font=self.button_font, bg="#f44336", fg="white")
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(root, text="Status: Not Running", font=self.button_font, bg="#f0f0f0")
        self.status_label.pack(pady=10)

        self.video_label = tk.Label(root, bg="#f0f0f0")
        self.video_label.pack()

        self.results_text = tk.Text(root, height=10, width=80, font=("Helvetica", 12), bg="#ffffff", fg="#000000", wrap=tk.WORD)
        self.results_text.pack(pady=10)
        self.results_text.insert(tk.END, "Detection Results:\n")
        self.detecting = False
        self.stop_event = threading.Event()

    def start_detection(self):
        if not self.detecting:
            self.detecting = True
            self.status_label.config(text="Status: Running")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.detect_face_in_real_time)
            self.thread.start()

    def stop_detection(self):
        if self.detecting:
            self.detecting = False
            self.stop_event.set()
            self.status_label.config(text="Status: Not Running")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.thread.join()

    def detect_face_in_real_time(self):
        known_faces_dir = "known"
        known_encodings, known_names = load_known_faces(known_faces_dir)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        output_dir = "captured_images"
        os.makedirs(output_dir, exist_ok=True)
        image_count = 0
        no_face_count = 0  
        max_no_face_count = 30  

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:
                no_face_count = 0  

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    name = "Unknown"

                    face_width = right - left
                    if face_width > 0:
                        distance = 640 / (face_width * 0.5)
                        print(f"Distance to the detected face: {distance:.2f} cm")

                        if distance < 5:  
                            speak_warning()

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]

                        if messagebox.askyesno("Open Door", f"Known person detected: {name}. Do you want to open the door?"):
                            self.open_door()
                            self.update_results(f"Door opened for: {name}")

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if name == "Unknown":
                        image_count += 1
                        image_name = os.path.join(output_dir, f'Unknown_{image_count}.jpg')
                        cv2.imwrite(image_name, frame)

                        # Analyze emotion for each unknown face
                        emotion_analysis = analyze_emotion(frame)
                        if emotion_analysis:
                            dominant_emotion = emotion_analysis[0]['dominant_emotion']
                            print(f"Detected emotion: {dominant_emotion}")

                            cv2.putText(frame, f'Emotion: {dominant_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                            self.update_results(f"Emotion detected: {dominant_emotion}")

                        # Analyze if the unknown person is a thief
                        thief_detected = analyze_thief(image_name) 
                        if thief_detected:
                            send_email_alert(image_name)

            else:
                no_face_count += 1  

                if no_face_count > max_no_face_count:
                    print("Warning: No face detected for a prolonged period.")
                    image_count += 1
                    image_name = os.path.join(output_dir, f'Unknown_{image_count}.jpg')
                    cv2.imwrite(image_name, frame)
                    thief_detected = analyze_thief(image_name)
                    if thief_detected:
                        print("Thief detected! Sending email alert...")
                        send_email_alert(image_name)
                    else:
                        print("No thief detected in the captured image.")     

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        cap.release()
        cv2.destroyAllWindows()

    def update_results(self, message):
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.yview(tk.END)

    def open_door(self):
        print("Door opened.")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
