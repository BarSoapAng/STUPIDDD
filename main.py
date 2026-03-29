from __future__ import annotations

import cv2
import pygame

from audio_engine import AudioEngine
from exporter import Exporter
from fx_engine import FXEngine
from pose_engine import PoseEngine


WINDOW_NAME = "DAB DETECTOR 9000"


def _scale_frame_to_screen(frame, screen_size: tuple[int, int]):
    screen_w, screen_h = screen_size
    frame_h, frame_w = frame.shape[:2]
    if screen_w <= 0 or screen_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return frame

    # Scale to cover the full screen, then crop the overflow from the center.
    scale = max(screen_w / frame_w, screen_h / frame_h)
    resized_w = max(screen_w, int(round(frame_w * scale)))
    resized_h = max(screen_h, int(round(frame_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    x0 = max(0, (resized_w - screen_w) // 2)
    y0 = max(0, (resized_h - screen_h) // 2)
    return resized[y0 : y0 + screen_h, x0 : x0 + screen_w]


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[camera] Could not open webcam.")
        return

    pygame.init()
    display_info = pygame.display.Info()
    screen_size = (
        max(1, display_info.current_w),
        max(1, display_info.current_h),
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_size[1])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    pose = PoseEngine(debug=True, debug_every=1)
    fx = FXEngine()
    audio = AudioEngine()
    exporter = Exporter()

    state = "IDLE"
    hold_counter = 0
    dab_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            pose_result = pose.classify(frame)

            if state == "IDLE":
                if pose_result.is_dab:
                    hold_counter += 1
                    if hold_counter >= 2:
                        state = "DAB"
                        dab_counter = 0
                        exporter.reset()
                        audio.play()
                else:
                    hold_counter = 0

            elif state == "DAB":
                dab_counter += 1
                frame = fx.apply(frame, pose_result, dab_counter)
                saved_name = exporter.maybe_export(frame, dab_counter)
                if saved_name:
                    fx.notify_saved()

                if not pose_result.is_dab:
                    state = "IDLE"
                    hold_counter = 0
                    dab_counter = 0
                    audio.stop()
                    fx.reset()

            bottom_text = "[Bottom Text]"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                bottom_text, font, font_scale, thickness
            )
            frame_h, frame_w = frame.shape[:2]
            text_x = max(0, (frame_w - text_w) // 2)
            text_y = max(text_h, frame_h - baseline - 12)

            cv2.putText(
                frame,
                bottom_text,
                (text_x, text_y),
                font,
                font_scale,
                (120, 120, 120),
                thickness,
            )

            display_frame = _scale_frame_to_screen(frame, screen_size)
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
