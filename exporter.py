from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np


class Exporter:
    def __init__(self) -> None:
        self.saved = False

    def reset(self) -> None:
        self.saved = False

    def maybe_export(self, frame: np.ndarray, dab_frame_count: int) -> str | None:
        if dab_frame_count == 10 and not self.saved:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dab_{timestamp}.jpg"
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"[export] Saved -> {filename}")
            self.saved = True
            return filename
        return None
