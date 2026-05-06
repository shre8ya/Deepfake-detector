import cv2
from deepfake_model import predict_frame

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fake_scores = []
    frame_skip = 10
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if frame_number % frame_skip != 0:
            continue

        score = predict_frame(frame)
        fake_scores.append(score)

    cap.release()

    # Handle empty case
    if len(fake_scores) == 0:
        return {
            "label": "ERROR",
            "confidence": 0
        }

    avg_score = sum(fake_scores) / len(fake_scores)

    label = "FAKE" if avg_score > 0.5 else "REAL"

    return {
        "label": label,
        "confidence": round(avg_score, 3)
    }