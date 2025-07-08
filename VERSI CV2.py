import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import pygame
from datetime import datetime
from ultralytics import YOLO
import os
from threading import Thread, Lock

class MultiCameraDetector:
    def __init__(self, num_cams=0):
        self.num_cams = num_cams
        self.captures = [cv2.VideoCapture(i) for i in range(num_cams)]
        self.frames = [None] * num_cams
        self.model = YOLO(r"D:\best.pt")  # Ganti sesuai path model Anda
        self.lock_apd = Lock()
        self.lock_api = Lock()

        try:
            pygame.mixer.init()
            self.sound_apd = pygame.mixer.Sound(r"D:\SMARTCAM SHIPYARD\ALARM KEBAKARAN.wav")
            self.sound_api = pygame.mixer.Sound(r"D:\SMARTCAM SHIPYARD\VOICE ALERT.wav")
            print("[AUDIO] Suara berhasil dimuat.")
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
            self.sound_apd = None
            self.sound_api = None

    def play_sound_once(self, sound, lock):
        if sound is None:
            print("[AUDIO] Suara tidak tersedia.")
            return
        if lock.locked():
            return
        def sound_thread():
            with lock:
                try:
                    pygame.mixer.stop()
                    sound.play()
                    time.sleep(sound.get_length())  # Tunggu suara selesai
                except Exception as e:
                    print(f"[AUDIO ERROR] {e}")
        Thread(target=sound_thread, daemon=True).start()

    def save_violation(self, frame, cam_id, violation_type):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cam{cam_id}_{violation_type}_{now}.jpg"
        path = f"static/violations/images/{filename}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, frame)
        print(f"[SAVE] Disimpan: {path}")
        try:
            generate_report(f"CAM-{cam_id}", path, violation_type)
        except Exception as e:
            print(f"[REPORT ERROR] {e}")

    def process_frame(self, cam_id):
        cap = self.captures[cam_id]
        if not cap.isOpened():
            print(f"[ERROR] Kamera {cam_id} gagal dibuka.")
            return

        print(f"[INFO] Kamera {cam_id} aktif.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] Kamera {cam_id} tidak memberi frame.")
                time.sleep(1)
                continue

            results = self.model(frame, verbose=False)[0]
            detected = [self.model.names[int(cls)] for cls in results.boxes.cls]
            boxes = results.boxes.xyxy.cpu().numpy()
            print(f"[DEBUG] Kamera {cam_id} deteksi: {detected}")

            # Deteksi api
            fires = [boxes[i] for i, name in enumerate(detected) if name == 'fire']
            if fires:
                self.play_sound_once(self.sound_api, self.lock_api)
                self.save_violation(frame, cam_id, "fire")

            # Deteksi APD
            for i, name in enumerate(detected):
                if name == 'person':
                    x1, y1, x2, y2 = boxes[i]
                    items_inside = [
                        detected[j] for j in range(len(boxes))
                        if detected[j] in ['helmet', 'gloves', 'shoes', 'wearpack']
                        and x1 < boxes[j][0] < x2 and y1 < boxes[j][1] < y2
                    ]
                    if not {'helmet', 'gloves', 'shoes', 'wearpack'}.issubset(set(items_inside)):
                        self.play_sound_once(self.sound_apd, self.lock_apd)
                        self.save_violation(frame, cam_id, "incomplete_apd")

            annotated_frame = results.plot()
            self.frames[cam_id] = annotated_frame

            # Tampilkan frame ke layar
            cv2.imshow(f'Camera {cam_id}', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(f'Camera {cam_id}')

    def run(self):
        for cam_id in range(self.num_cams):
            thread = Thread(target=self.process_frame, args=(cam_id,), daemon=True)
            thread.start()

        # Tunggu semua jendela ditutup
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[EXIT] Keluar dari aplikasi.")
                break
        for cap in self.captures:
            cap.release()
        cv2.destroyAllWindows()
