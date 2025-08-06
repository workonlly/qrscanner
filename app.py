from flask import Flask, render_template, Response
import cv2
from pyzbar.pyzbar import decode
import numpy as np
import os

app = Flask(__name__)

def gen_frames():
    last_detected_value = "Point camera at a QR or Barcode"
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        decoded_objects = decode(frame)

        if decoded_objects:
            for code in decoded_objects:
                code_data = code.data.decode('utf-8')
                code_type = code.type
                last_detected_value = f"Last Scan -> {code_type}: {code_data}"

                # Draw green bounding box
                if code.polygon:
                    points = np.array([code.polygon], np.int32)
                    points = points.reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], True, (0, 255, 0), 3)

                # Draw code data above box
                rect = code.rect
                text_position = (rect.left, max(rect.top - 10, 20))
                cv2.putText(frame, code_data, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Persistent bar at the bottom
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, last_detected_value, (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Encode to JPEG and yield as MJPEG frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
