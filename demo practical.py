import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import pygame
import os
from datetime import datetime
from ultralytics import YOLO

# Inisialisasi model YOLO
model = YOLO(r"D:\best.pt")

# Inisialisasi suara
pygame.mixer.init()
fire_sound = pygame.mixer.Sound(r"D:\SMARTCAM SHIPYARD\ALARM KEBAKARAN.wav")
apd_sound = pygame.mixer.Sound(r"D:\SMARTCAM SHIPYARD\VOICE ALERT.wav")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Pastikan direktori untuk log dan capture ada
os.makedirs("static/capture", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Fungsi utilitas
def check_apd_compliance(detected):
    apd_items = ['Helm', 'Wearpack', 'Gloves', 'Shoes']
    person_detected = 'Person' in detected
    apd_worn = all(item in detected for item in apd_items)

    if person_detected and not apd_worn:
        return "APD Tidak Lengkap"
    return "Lengkap"

def log_detection(event, frame, timestamp):
    filename = f"static/capture/{event}_{timestamp.replace(':', '-')}.jpg"
    cv2.imwrite(filename, frame)
    with open("logs/detection_log.txt", "a") as f:
        f.write(f"[{timestamp}] - {event}\n")

def play_sound(sound):
    if not pygame.mixer.get_busy():
        sound.play()

# Main loop
while True:
    success, frame = cap.read()
    if not success:
        print("Gagal membaca frame dari kamera")
        break
    
    # Deteksi objek
    results = model(frame)[0]
    names = model.names
    detected = [names[int(cls)] for cls in results.boxes.cls]
    
    # Timestamp dan status APD
    frame_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = check_apd_compliance(detected)
    
    # Penanganan event
    if 'Api' in detected:
        play_sound(fire_sound)
        log_detection("Api", frame, frame_time)
        cv2.putText(frame, "KEBAKARAN TERDETEKSI!", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    elif status == "APD Tidak Lengkap":
        play_sound(apd_sound)
        log_detection("APD Tidak Lengkap", frame, frame_time)
        cv2.putText(frame, "APD TIDAK LENGKAP!", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Gambar bounding box
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Tampilkan status
    cv2.putText(frame, f"Status: {status}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, frame_time, (10, frame.shape[0]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Tampilkan frame
    cv2.imshow('SmartCam Shipyard', frame)
    
    # Keluar dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()