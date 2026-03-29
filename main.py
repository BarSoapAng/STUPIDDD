from __future__ import annotations

import cv2
import pygame

from audio_engine import AudioEngine
from exporter import Exporter
from fx_engine import FXEngine
from pose_engine import PoseEngine


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[camera] Could not open webcam.")
        return

    pygame.init()

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
                    if hold_counter >= 8:
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

            color = (0, 255, 80) if state == "DAB" else (180, 180, 180)
            cv2.putText(
                frame,
                f"STATE: {state}",
                (12, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"FRAME: {dab_counter}",
                (12, 56),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                color,
                1,
            )
            cv2.putText(
                frame,
                f"HOLD:  {hold_counter}",
                (12, 80),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (120, 120, 120),
                1,
            )

            cv2.imshow("DAB DETECTOR 9000", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
