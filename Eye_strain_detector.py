"""
Eye Strain Detector
- Press 'c' to calibrate (stand at comfortable reading distance and press c).
- Press 'q' to quit.
"""

import cv2
import time
import numpy as np
import mediapipe as mp

# Optional sound alert
try:
    import simpleaudio as sa
    SOUND_AVAILABLE = True
except Exception:
    SOUND_AVAILABLE = False

# -----------------------
# PARAMETERS (tweak if needed)
# -----------------------
CALIBRATE_KEY = ord('c')
QUIT_KEY = ord('q')

# factor by which pixel-IPD must increase to consider "too close"
CLOSE_FACTOR = 1.35   # warn if current_ipd > baseline_ipd * CLOSE_FACTOR

# how many seconds of continuous "too close" triggers warning
SUSTAIN_SECONDS = 5.0

# whether to play beep on warning (requires simpleaudio)
BEEP_ON_WARNING = True

# beep params (440Hz tone for 0.4s)
def play_beep(duration=0.4, freq=440):
    if not SOUND_AVAILABLE or not BEEP_ON_WARNING:
        return
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = np.sin(freq * t * 2 * np.pi)
    audio = tone * (2**15 - 1) / np.max(np.abs(tone))
    audio = audio.astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    # don't block; let it play

# -----------------------
# Setup MediaPipe Face Mesh
# -----------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# helper: compute pixel distance between two normalized landmarks
def landmark_pixel_distance(lm1, lm2, image_w, image_h):
    x1, y1 = int(lm1.x * image_w), int(lm1.y * image_h)
    x2, y2 = int(lm2.x * image_w), int(lm2.y * image_h)
    return np.hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)

# -----------------------
# Main loop
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit(1)

baseline_ipd = None
too_close_start = None
warning_shown = False

print("Eye Strain Detector")
print(" - Press 'c' to calibrate at a comfortable reading distance.")
print(" - Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror for natural UX
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)
    info_text = ""

    if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
        lm = results.multi_face_landmarks[0].landmark

        # MediaPipe landmarks for approximate eye centers / outer corners:
        # 33 = left eye outer corner, 133 = left eye inner corner (approx pupil region)
        # 362 = right eye outer corner, 263 = right eye inner corner
        # For a simple IPD proxy we'll use landmarks 33 and 263 (outer->outer)
        lm_left = lm[33]
        lm_right = lm[263]

        ipd_px, p1, p2 = landmark_pixel_distance(lm_left, lm_right, w, h)
        # draw line between eyes
        cv2.line(frame, p1, p2, (200, 200, 0), 2)
        cv2.circle(frame, p1, 3, (0,255,0), -1)
        cv2.circle(frame, p2, 3, (0,255,0), -1)

        # Show measured IPD
        cv2.putText(frame, f"IPD(px): {int(ipd_px)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Calibration
        key = cv2.waitKey(1) & 0xFF
        if key == CALIBRATE_KEY:
            baseline_ipd = ipd_px
            too_close_start = None
            warning_shown = False
            print(f"Calibrated baseline IPD to {int(baseline_ipd)} px")

        # If not calibrated, prompt user
        if baseline_ipd is None:
            info_text = "Press 'c' to calibrate at comfortable reading distance"
            cv2.putText(frame, info_text, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            # Check if current ipd indicates too close
            threshold = baseline_ipd * CLOSE_FACTOR
            if ipd_px > threshold:
                # entering too-close state
                if too_close_start is None:
                    too_close_start = time.time()
                    warning_shown = False
                else:
                    elapsed = time.time() - too_close_start
                    remaining = max(0, SUSTAIN_SECONDS - elapsed)
                    cv2.putText(frame, f"Too close! Hold for {remaining:.1f}s to warn",
                                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if elapsed >= SUSTAIN_SECONDS and not warning_shown:
                        # show warning
                        warning_text = "MOVE BACK - Protect your eyes 👀"
                        cv2.putText(frame, warning_text, (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print("WARNING: Move back from the screen.")
                        if BEEP_ON_WARNING:
                            play_beep()
                        warning_shown = True
            else:
                # reset
                too_close_start = None
                warning_shown = False
                cv2.putText(frame, f"Distance OK", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    else:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    cv2.imshow("Eye Strain Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == QUIT_KEY:
        break
    # allow calibration via keyboard outside loop read
    if key == CALIBRATE_KEY:
        # handled earlier; keep for reliability
        pass

cap.release()
cv2.destroyAllWindows()
