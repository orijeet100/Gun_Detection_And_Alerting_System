import cv2
import imutils
import datetime

# Load the gun cascade classifier
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Open the input video file
input_video_path = 'input_video.mp4'
output_video_path = 'output_video.mp4'
input_video = cv2.VideoCapture(input_video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))

# Process each frame of the input video
gun_exist = False
while True:
    ret, frame = input_video.read()
    if not ret:
        break

    # Resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect guns in the frame
    guns = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))

    # Draw rectangles around the detected guns
    if len(guns) > 0:
        gun_exist = True
    for (x, y, w, h) in guns:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Add timestamp to the frame
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    # Write the frame to the output video
    output_video.write(frame)

    # Display the frame
    cv2.imshow("Gun Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the input and output video objects
input_video.release()
output_video.release()
cv2.destroyAllWindows()

# Print message if guns were detected
if gun_exist:
    print("Guns detected. Output video saved as 'output_video.mp4'")
else:
    print("No guns detected.")




