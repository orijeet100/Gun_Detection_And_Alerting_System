import cv2
import os
import time
from collections import deque
from ultralytics import YOLO
from twilio.rest import Client

# Twilio credentials
# Twilio credentials
TWILIO_ACCOUNT_SID = 'AC758b0acfd73c70151db674f59af10e80'
TWILIO_AUTH_TOKEN = '6656b5a90ba67f98032a8961b6c2e68f'
TWILIO_FROM_NUMBER = '+18443354341'  # your Twilio number
TO_PHONE_NUMBER = '+16173316848'     # your phone number (verified)

def trigger_twilio_call():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml='<Response><Say>Gun detected on the surveillance feed. Please check immediately.</Say></Response>',
            to=TO_PHONE_NUMBER,
            from_=TWILIO_FROM_NUMBER
        )
        print(f"Call initiated: {call.sid}")
    except Exception as e:
        print("Failed to initiate call:", e)


def send_twilio_sms():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="🚨 Gun detected on the surveillance feed. Please check immediately.",
            from_=TWILIO_FROM_NUMBER,
            to=TO_PHONE_NUMBER
        )
        print(f"💬 SMS sent: {message.sid}")
    except Exception as e:
        print("❌ SMS failed:", e)


# === Load YOLOv8 model ===
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# === Setup webcam capture ===
cap = cv2.VideoCapture(0)  # For Windows systems with DroidCam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# === Create detections folder if not exist ===
os.makedirs("detections", exist_ok=True)

# === Frame buffer to store pre-detection context ===
buffer = deque(maxlen=20)
save_frames = 5
instance_counter = len([d for d in os.listdir("detections") if d.startswith("instance_")])

# === Post-detection frame saving state ===
post_buffer = []
post_saving = False
post_countdown = 0

# === Detection timing management ===
last_detection_time = 0
min_interval = 5  # in seconds

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    current_time = time.time()
    buffer.append(frame.copy())

    # === Detection ===
    results = model(frame, stream=True)
    gun_detected = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence > 0.6:
                gun_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Gun {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === If gun is detected and cooldown period passed ===
    if gun_detected:
        if (current_time - last_detection_time > min_interval) and not post_saving:
            last_detection_time = current_time
            instance_counter += 1
            instance_path = f"detections/instance_{instance_counter}"
            os.makedirs(instance_path, exist_ok=True)

            # Save last 5 frames
            buffer_list = list(buffer)[-save_frames:]
            for idx, f in enumerate(buffer_list):
                cv2.imwrite(f"{instance_path}/frame_pre_{idx:02d}.jpg", f)

            # Save detected frame
            cv2.imwrite(f"{instance_path}/frame_detected.jpg", frame)

            # Start saving next 5 frames
            post_saving = True
            post_countdown = save_frames
            post_buffer = []
            trigger_twilio_call()
            send_twilio_sms()
            print("Detection Happened")

    elif post_saving:
        post_buffer.append(frame.copy())
        post_countdown -= 1

        if post_countdown == 0:
            for idx, f in enumerate(post_buffer):
                cv2.imwrite(f"{instance_path}/frame_post_{idx:02d}.jpg", f)
            post_saving = False
            post_buffer.clear()
            buffer.clear()

    # === Display ===
    cv2.imshow("Gun Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
