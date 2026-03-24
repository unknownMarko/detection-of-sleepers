"""
Sleeper Detection System
========================
Detects faces and eyes using OpenCV Haar cascades from webcam or video file.
If eyes were previously detected but are not seen for 5 seconds,
the program captures a "BUSTED!" screenshot with timestamp.

BONUS features:
  - Awake vs sleeping time tracking
  - Awake/sleep ratio percentage
  - BUSTED counter
  - Smile detection
  - Warning flash before BUSTED trigger
  - FPS display
  - Session statistics on exit

Usage:
    python detector.py                         # webcam (default)
    python detector.py --source video.mp4      # video file
    python detector.py --source 1              # specific camera index
"""

import cv2
import os
import sys
import time
import argparse
from datetime import datetime


# ============================================================================
# Configuration
# ============================================================================

EYES_CLOSED_THRESHOLD = 5.0  # seconds without eyes detected -> BUSTED

# Haar cascade detectMultiScale parameters
FACE_SCALE_FACTOR = 1.3
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (30, 30)

EYE_SCALE_FACTOR = 1.1
EYE_MIN_NEIGHBORS = 7
EYE_MIN_SIZE = (20, 20)

SMILE_SCALE_FACTOR = 1.8
SMILE_MIN_NEIGHBORS = 20
SMILE_MIN_SIZE = (25, 25)

# Colors (BGR format)
COLOR_FACE_RECT = (255, 200, 0)  # cyan-ish
COLOR_EYE_RECT = (0, 255, 0)  # green
COLOR_SMILE_RECT = (255, 0, 255)  # magenta
COLOR_WARNING = (0, 0, 255)  # red
COLOR_OK = (0, 200, 0)  # green
COLOR_STATUS = (0, 255, 255)  # yellow
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_PANEL_BG = (30, 30, 30)

SCREENSHOT_DIR = "screenshots"


# ============================================================================
# Haar Cascade Loading
# ============================================================================


def load_cascades():
    """Load Haar cascade classifiers for face, eye, and smile detection."""
    cascade_dir = cv2.data.haarcascades

    face_cascade = cv2.CascadeClassifier(
        os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
    )
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(cascade_dir, "haarcascade_eye.xml")
    )
    smile_cascade = cv2.CascadeClassifier(
        os.path.join(cascade_dir, "haarcascade_smile.xml")
    )

    if face_cascade.empty():
        print("[ERROR] Failed to load face cascade.")
        sys.exit(1)
    if eye_cascade.empty():
        print("[ERROR] Failed to load eye cascade.")
        sys.exit(1)
    if smile_cascade.empty():
        print("[WARN] Smile cascade not loaded — smile detection disabled.")
        smile_cascade = None

    print("[INFO] Haar cascades loaded successfully.")
    return face_cascade, eye_cascade, smile_cascade


# ============================================================================
# Detection Functions
# ============================================================================


def detect_faces(frame_gray, face_cascade):
    """Detect faces in a grayscale frame. Returns list of (x, y, w, h)."""
    return face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )


def detect_eyes(roi_gray, eye_cascade):
    """Detect eyes within a face ROI. Returns list of (x, y, w, h)."""
    return eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=EYE_SCALE_FACTOR,
        minNeighbors=EYE_MIN_NEIGHBORS,
        minSize=EYE_MIN_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )


def detect_smile(roi_gray, smile_cascade):
    """Detect smile within a face ROI. Returns list of (x, y, w, h)."""
    if smile_cascade is None:
        return []
    return smile_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=SMILE_SCALE_FACTOR,
        minNeighbors=SMILE_MIN_NEIGHBORS,
        minSize=SMILE_MIN_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )


# ============================================================================
# Drawing Helpers
# ============================================================================


def draw_text_with_bg(
    frame,
    text,
    origin,
    font_scale=0.6,
    color=COLOR_WHITE,
    bg_color=COLOR_BLACK,
    thickness=1,
    padding=5,
):
    """Draw text with an opaque background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_detections(frame, faces, all_eyes, all_smiles, face_offsets):
    """Draw rectangles around detected faces, eyes, and smiles."""
    for i, (fx, fy, fw, fh) in enumerate(faces):
        # Face rectangle
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), COLOR_FACE_RECT, 2)

        # Eyes (coordinates are relative to the eye-search ROI)
        if i < len(all_eyes):
            ey_offset = face_offsets[i]  # y-offset of eye search region
            for ex, ey, ew, eh in all_eyes[i]:
                abs_x = fx + ex
                abs_y = fy + ey_offset + ey
                cv2.rectangle(
                    frame, (abs_x, abs_y), (abs_x + ew, abs_y + eh), COLOR_EYE_RECT, 2
                )

        # Smile (coordinates are relative to lower face ROI)
        if i < len(all_smiles):
            for sx, sy, sw, sh in all_smiles[i]:
                abs_x = fx + sx
                abs_y = fy + fh // 2 + sy
                cv2.rectangle(
                    frame, (abs_x, abs_y), (abs_x + sw, abs_y + sh), COLOR_SMILE_RECT, 2
                )


def draw_status_panel(frame, state, fps):
    """Draw the top status panel with detection info."""
    h, w = frame.shape[:2]

    # --- Top panel ---
    panel_h = 95
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), COLOR_PANEL_BG, cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, panel_h), (w, panel_h), (80, 80, 80), 1)

    # FPS
    draw_text_with_bg(
        frame,
        f"FPS: {fps:.1f}",
        (10, 20),
        font_scale=0.5,
        color=COLOR_OK,
        bg_color=COLOR_PANEL_BG,
    )

    # Faces
    draw_text_with_bg(
        frame,
        f"Faces: {state['face_count']}",
        (130, 20),
        font_scale=0.5,
        color=COLOR_WHITE,
        bg_color=COLOR_PANEL_BG,
    )

    # Eyes status
    if state["eyes_detected"]:
        eyes_txt = "Eyes: OPEN"
        eyes_clr = COLOR_OK
    else:
        eyes_txt = "Eyes: NOT DETECTED"
        eyes_clr = COLOR_WARNING
    draw_text_with_bg(
        frame,
        eyes_txt,
        (10, 45),
        font_scale=0.5,
        color=eyes_clr,
        bg_color=COLOR_PANEL_BG,
    )

    # Smile
    if state["smile_detected"]:
        draw_text_with_bg(
            frame,
            "Smile: YES :)",
            (250, 45),
            font_scale=0.5,
            color=COLOR_SMILE_RECT,
            bg_color=COLOR_PANEL_BG,
        )

    # Countdown timer (eyes were seen before but not now)
    if (
        state["eyes_ever_seen"]
        and not state["eyes_detected"]
        and not state["busted_active"]
    ):
        elapsed = time.time() - state["last_eyes_time"]
        remaining = max(0.0, EYES_CLOSED_THRESHOLD - elapsed)
        timer_clr = COLOR_WARNING if remaining < 2.0 else COLOR_STATUS
        draw_text_with_bg(
            frame,
            f"Eyes closed: {elapsed:.1f}s / {EYES_CLOSED_THRESHOLD:.0f}s",
            (10, 70),
            font_scale=0.5,
            color=timer_clr,
            bg_color=COLOR_PANEL_BG,
        )

    # --- Bottom stats panel ---
    panel_bot_h = 55
    bot_y = h - panel_bot_h
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, bot_y), (w, h), COLOR_PANEL_BG, cv2.FILLED)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, bot_y), (w, bot_y), (80, 80, 80), 1)

    draw_text_with_bg(
        frame,
        f"Awake: {state['awake_time']:.1f}s",
        (10, bot_y + 20),
        font_scale=0.4,
        color=COLOR_OK,
        bg_color=COLOR_PANEL_BG,
    )
    draw_text_with_bg(
        frame,
        f"Sleeping: {state['sleep_time']:.1f}s",
        (180, bot_y + 20),
        font_scale=0.4,
        color=COLOR_WARNING,
        bg_color=COLOR_PANEL_BG,
    )
    draw_text_with_bg(
        frame,
        f"Awake ratio: {state['awake_ratio']:.0f}%",
        (10, bot_y + 43),
        font_scale=0.4,
        color=COLOR_STATUS,
        bg_color=COLOR_PANEL_BG,
    )
    draw_text_with_bg(
        frame,
        f"BUSTED count: {state['busted_count']}",
        (180, bot_y + 43),
        font_scale=0.4,
        color=COLOR_WARNING,
        bg_color=COLOR_PANEL_BG,
    )

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw_text_with_bg(
        frame,
        ts,
        (w - 195, bot_y + 20),
        font_scale=0.4,
        color=(180, 180, 180),
        bg_color=COLOR_PANEL_BG,
    )


# ============================================================================
# BUSTED! Screenshot
# ============================================================================


def save_busted_screenshot(frame):
    """Save a screenshot with large BUSTED! overlay and timestamp."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    shot = frame.copy()
    h, w = shot.shape[:2]

    now = datetime.now()
    timestamp_display = now.strftime("%Y-%m-%d %H:%M:%S")
    filename = now.strftime("BUSTED_%Y%m%d_%H%M%S.png")
    filepath = os.path.join(SCREENSHOT_DIR, filename)

    # --- BUSTED! text (centered, large) ---
    font = cv2.FONT_HERSHEY_DUPLEX
    busted_text = "BUSTED!"
    scale = 3.0
    thick = 5
    (tw, th), baseline = cv2.getTextSize(busted_text, font, scale, thick)
    tx = (w - tw) // 2
    ty = (h + th) // 2

    pad = 25
    # Background box
    cv2.rectangle(
        shot,
        (tx - pad, ty - th - pad),
        (tx + tw + pad, ty + baseline + pad),
        COLOR_BLACK,
        cv2.FILLED,
    )
    cv2.rectangle(
        shot,
        (tx - pad, ty - th - pad),
        (tx + tw + pad, ty + baseline + pad),
        COLOR_WARNING,
        3,
    )
    # BUSTED! text
    cv2.putText(
        shot, busted_text, (tx, ty), font, scale, COLOR_WARNING, thick, cv2.LINE_AA
    )

    # --- Timestamp below ---
    ts_scale = 0.9
    ts_thick = 2
    (tsw, tsh), _ = cv2.getTextSize(timestamp_display, font, ts_scale, ts_thick)
    tsx = (w - tsw) // 2
    tsy = ty + baseline + pad + tsh + 25
    cv2.rectangle(
        shot,
        (tsx - 10, tsy - tsh - 10),
        (tsx + tsw + 10, tsy + 10),
        COLOR_BLACK,
        cv2.FILLED,
    )
    cv2.putText(
        shot,
        timestamp_display,
        (tsx, tsy),
        font,
        ts_scale,
        COLOR_WHITE,
        ts_thick,
        cv2.LINE_AA,
    )

    cv2.imwrite(filepath, shot)
    print(f"[BUSTED!] Screenshot saved: {filepath}")
    return filepath


# ============================================================================
# Video Source
# ============================================================================


def open_video_source(source_str):
    """
    Open video capture from camera index or video file path.
    source_str: "0", "1", ... for camera, or file path for video.
    """
    # Try as camera index
    try:
        idx = int(source_str)
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None:
                print(f"[INFO] Opened camera index {idx}")
                return cap
        cap.release()
        print(f"[WARN] Camera index {idx} not available.")
    except ValueError:
        pass

    # Try as file path
    if os.path.isfile(source_str):
        cap = cv2.VideoCapture(source_str)
        if cap.isOpened():
            print(f"[INFO] Opened video file: {source_str}")
            return cap
        cap.release()
        print(f"[ERROR] Cannot open video file: {source_str}")
    else:
        print(f"[ERROR] File not found: {source_str}")

    return None


# ============================================================================
# CLI Arguments
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sleeper Detection - Haar cascade face & eye detection with drowsiness alert"
    )
    parser.add_argument(
        "--source",
        "-s",
        default="0",
        help="Video source: camera index (0,1,...) or path to video file (default: 0)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=EYES_CLOSED_THRESHOLD,
        help=f"Seconds without eyes before BUSTED (default: {EYES_CLOSED_THRESHOLD})",
    )
    parser.add_argument(
        "--no-smile",
        action="store_true",
        help="Disable smile detection (BONUS feature)",
    )
    return parser.parse_args()


# ============================================================================
# Main Loop
# ============================================================================


def main():
    args = parse_args()

    global EYES_CLOSED_THRESHOLD
    EYES_CLOSED_THRESHOLD = args.threshold

    print("=" * 55)
    print("   SLEEPER DETECTION SYSTEM")
    print("   OpenCV Haar Cascade Face & Eye Detection")
    print("=" * 55)

    # Load cascades
    face_cascade, eye_cascade, smile_cascade = load_cascades()
    if args.no_smile:
        smile_cascade = None

    # Open video
    cap = open_video_source(args.source)
    if cap is None:
        print("[ERROR] No video source available. Exiting.")
        sys.exit(1)

    # State
    state = {
        "eyes_detected": False,
        "eyes_ever_seen": False,
        "last_eyes_time": time.time(),
        "busted_active": False,  # True after BUSTED triggered, reset on eyes re-detected
        "busted_count": 0,
        "face_count": 0,
        "smile_detected": False,
        "awake_time": 0.0,
        "sleep_time": 0.0,
        "awake_ratio": 100.0,
    }

    # FPS tracking
    prev_time = time.time()
    fps_buf = []
    last_frame_time = time.time()

    window_name = "Sleeper Detection"
    print(f"\n[INFO] Detection running. Press 'q' or ESC to exit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Video file ended -> loop back
                try:
                    int(args.source)
                    # It was a camera -> real failure
                    print("[ERROR] Camera read failed.")
                    break
                except ValueError:
                    # Video file -> rewind
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now

            # FPS (smoothed over 30 frames)
            fps_buf.append(1.0 / dt if dt > 0 else 0.0)
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            fps = sum(fps_buf) / len(fps_buf)

            # ---- Grayscale + histogram equalization ----
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # ---- Face detection ----
            faces = detect_faces(gray, face_cascade)
            state["face_count"] = len(faces)

            eyes_found_this_frame = False
            smile_found_this_frame = False
            all_eyes = []
            all_smiles = []
            eye_y_offsets = []

            for fx, fy, fw, fh in faces:
                # Search eyes in upper 65% of face to avoid mouth false positives
                eye_region_h = int(fh * 0.65)
                roi_eyes = gray[fy : fy + eye_region_h, fx : fx + fw]
                eyes = detect_eyes(roi_eyes, eye_cascade)
                all_eyes.append(eyes)
                eye_y_offsets.append(0)  # offset within face is 0 (starts at fy)

                if len(eyes) >= 2:
                    eyes_found_this_frame = True

                # Smile in lower 50% of face
                roi_smile = gray[fy + fh // 2 : fy + fh, fx : fx + fw]
                smiles = detect_smile(roi_smile, smile_cascade)
                all_smiles.append(smiles)
                if len(smiles) > 0:
                    smile_found_this_frame = True

            # ---- State update ----
            state["smile_detected"] = smile_found_this_frame

            if eyes_found_this_frame:
                state["eyes_detected"] = True
                state["eyes_ever_seen"] = True
                state["last_eyes_time"] = now
                state["busted_active"] = False
                state["awake_time"] += dt
            else:
                state["eyes_detected"] = False
                if state["eyes_ever_seen"]:
                    state["sleep_time"] += dt

            # Awake ratio
            total = state["awake_time"] + state["sleep_time"]
            state["awake_ratio"] = (
                (state["awake_time"] / total * 100) if total > 0 else 100.0
            )

            # ---- BUSTED! check ----
            if (
                state["eyes_ever_seen"]
                and not state["eyes_detected"]
                and not state["busted_active"]
            ):
                elapsed = now - state["last_eyes_time"]
                if elapsed >= EYES_CLOSED_THRESHOLD:
                    save_busted_screenshot(frame)
                    state["busted_active"] = True
                    state["busted_count"] += 1

            # ---- Draw everything ----
            draw_detections(frame, faces, all_eyes, all_smiles, eye_y_offsets)
            draw_status_panel(frame, state, fps)

            # Warning flash: red border when close to BUSTED
            if (
                state["eyes_ever_seen"]
                and not state["eyes_detected"]
                and not state["busted_active"]
            ):
                elapsed = now - state["last_eyes_time"]
                if elapsed > EYES_CLOSED_THRESHOLD - 2.0:
                    if int(now * 4) % 2 == 0:
                        cv2.rectangle(
                            frame,
                            (0, 0),
                            (frame.shape[1] - 1, frame.shape[0] - 1),
                            COLOR_WARNING,
                            4,
                        )

            # ---- Display ----
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("s"):
                # Manual screenshot
                os.makedirs(SCREENSHOT_DIR, exist_ok=True)
                manual_path = os.path.join(
                    SCREENSHOT_DIR, datetime.now().strftime("manual_%Y%m%d_%H%M%S.png")
                )
                cv2.imwrite(manual_path, frame)
                print(f"[INFO] Manual screenshot: {manual_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Session statistics
        print("\n" + "=" * 55)
        print("   SESSION STATISTICS")
        print("=" * 55)
        print(f"   Awake time:    {state['awake_time']:.1f}s")
        print(f"   Sleep time:    {state['sleep_time']:.1f}s")
        print(f"   Awake ratio:   {state['awake_ratio']:.1f}%")
        print(f"   BUSTED count:  {state['busted_count']}")
        print("=" * 55)


if __name__ == "__main__":
    main()
