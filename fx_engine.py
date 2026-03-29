from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pose_engine import PoseResult


@dataclass
class ActiveSprite:
    image: np.ndarray  # RGBA
    x: int
    y: int
    scale: float
    age: int


class FXEngine:
    def __init__(self, assets_dir: str = "assets") -> None:
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        self.sunglasses_rgba = self._load_or_create_sunglasses()
        self.sprite_assets = self._load_mlg_sprites()

        self.active_sprites: list[ActiveSprite] = []
        self.rainbow_base: np.ndarray | None = None
        self.hue_offset = 0
        self.save_notice_timer = 0

    def reset(self) -> None:
        self.active_sprites.clear()
        self.rainbow_base = None
        self.hue_offset = 0
        self.save_notice_timer = 0

    def notify_saved(self) -> None:
        self.save_notice_timer = 60

    def apply(self, frame: np.ndarray, pose_result: PoseResult, dab_frame_count: int) -> np.ndarray:
        if dab_frame_count == 1:
            self._on_dab_activated(frame, pose_result)

        if self.rainbow_base is None or self.rainbow_base.shape[:2] != frame.shape[:2]:
            self._build_rainbow(frame.shape[:2])

        out = frame.copy()
        out = self._apply_rainbow(out)
        self._update_and_draw_sprites(out, pose_result, dab_frame_count)
        out = self._apply_sunglasses(out, pose_result, dab_frame_count)
        self._draw_save_notice(out)
        return out

    def _on_dab_activated(self, frame: np.ndarray, pose_result: PoseResult) -> None:
        self.hue_offset = 0
        self._build_rainbow(frame.shape[:2])
        self.active_sprites.clear()
        spawn_count = random.randint(3, 5)
        for _ in range(spawn_count):
            self._spawn_sprite(frame.shape[:2], pose_result)

    def _load_or_create_sunglasses(self) -> np.ndarray:
        path = self.assets_dir / "sunglasses.png"
        if not path.exists():
            canvas = Image.new("RGBA", (220, 60), (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.ellipse((10, 8, 98, 52), fill=(0, 0, 0, 255))
            draw.ellipse((122, 8, 210, 52), fill=(0, 0, 0, 255))
            draw.rectangle((96, 28, 124, 32), fill=(0, 0, 0, 255))
            draw.rectangle((42, 18, 46, 22), fill=(255, 255, 255, 255))
            draw.rectangle((154, 18, 158, 22), fill=(255, 255, 255, 255))
            canvas.save(path)

        sunglasses = Image.open(path).convert("RGBA")
        return np.array(sunglasses)

    def _load_mlg_sprites(self) -> list[np.ndarray]:
        sprites: list[np.ndarray] = []
        sprite_paths = sorted(self.assets_dir.glob("mlg_sprite_*.png"))
        if not sprite_paths:
            sprite_paths = [
                path
                for path in sorted(self.assets_dir.glob("*.png"))
                if path.name.lower() != "sunglasses.png"
            ]

        for sprite_path in sprite_paths:
            sprite_bgra = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
            if sprite_bgra is None:
                continue
            if sprite_bgra.ndim != 3 or sprite_bgra.shape[2] < 4:
                continue
            sprite_rgba = cv2.cvtColor(sprite_bgra, cv2.COLOR_BGRA2RGBA)
            sprites.append(sprite_rgba)
        return sprites

    def _build_rainbow(self, shape_hw: tuple[int, int]) -> None:
        h, w = shape_hw
        hue = np.tile(np.linspace(0, 180, w, dtype=np.uint8), (h, 1))
        sat = np.full((h, w), 255, dtype=np.uint8)
        val = np.full((h, w), 200, dtype=np.uint8)
        hsv = np.stack([hue, sat, val], axis=2)
        self.rainbow_base = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_rainbow(self, frame: np.ndarray) -> np.ndarray:
        if self.rainbow_base is None:
            return frame
        self.hue_offset = (self.hue_offset + 2) % 180
        rainbow = np.roll(self.rainbow_base, self.hue_offset, axis=1)
        return cv2.addWeighted(frame, 0.6, rainbow, 0.4, 0.0)

    def _apply_sunglasses(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        dab_frame_count: int,
    ) -> np.ndarray:
        eye_left = pose_result.landmarks.get("LEFT_EYE")
        eye_right = pose_result.landmarks.get("RIGHT_EYE")
        if not eye_left or not eye_right:
            return frame

        inter_eye_dist = math.dist(eye_left, eye_right)
        if inter_eye_dist < 2:
            return frame

        target_w = max(1, int(inter_eye_dist * 2.2))
        src_h, src_w = self.sunglasses_rgba.shape[:2]
        target_h = max(1, int(target_w * (src_h / src_w)))
        resized = cv2.resize(self.sunglasses_rgba, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        angle = math.degrees(math.atan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0]))
        rotated = self._rotate_rgba(resized, angle)

        eye_center_x = int((eye_left[0] + eye_right[0]) * 0.5)
        eye_center_y = int((eye_left[1] + eye_right[1]) * 0.5)
        anim_t = min(dab_frame_count / 20.0, 1.0)
        eased = self._ease_out(anim_t)
        y = int(self._lerp(eye_center_y - 150, eye_center_y, eased))

        x0 = eye_center_x - rotated.shape[1] // 2
        y0 = y - rotated.shape[0] // 2
        self._copy_to_with_alpha_mask(frame, rotated, x0, y0)
        return frame

    def _update_and_draw_sprites(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        dab_frame_count: int,
    ) -> None:
        if self.sprite_assets and dab_frame_count % 40 == 0:
            self._spawn_sprite(frame.shape[:2], pose_result)

        survivors: list[ActiveSprite] = []
        for sprite in self.active_sprites:
            sprite.age += 1
            pop_t = min(sprite.age / 10.0, 1.0)
            sprite.scale = self._lerp(0.5, 1.2, self._ease_out(pop_t))

            opacity = min(1.0, sprite.age / 8.0)
            if sprite.age > 80:
                fade_t = min((sprite.age - 80) / 15.0, 1.0)
                opacity *= max(0.0, 1.0 - fade_t)

            if opacity <= 0.0:
                continue

            scaled = self._resize_rgba(sprite.image, sprite.scale)
            self._alpha_blend_rgba(frame, scaled, sprite.x, sprite.y, opacity)
            survivors.append(sprite)

        self.active_sprites = survivors

    def _spawn_sprite(self, frame_shape_hw: tuple[int, int], pose_result: PoseResult) -> None:
        if not self.sprite_assets:
            return

        frame_h, frame_w = frame_shape_hw
        sprite_image = random.choice(self.sprite_assets)
        sprite_h, sprite_w = sprite_image.shape[:2]

        face_box = None
        if pose_result.nose_px is not None:
            nx, ny = pose_result.nose_px
            face_box = (nx - 120, ny - 120, nx + 120, ny + 120)

        max_x = max(0, frame_w - sprite_w)
        max_y = max(0, frame_h - sprite_h)

        x, y = 0, 0
        for _ in range(30):
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
            if not self._intersects_face_box(x, y, sprite_w, sprite_h, face_box):
                break

        self.active_sprites.append(
            ActiveSprite(
                image=sprite_image,
                x=x,
                y=y,
                scale=0.5,
                age=0,
            )
        )

    @staticmethod
    def _intersects_face_box(
        x: int,
        y: int,
        w: int,
        h: int,
        face_box: tuple[int, int, int, int] | None,
    ) -> bool:
        if face_box is None:
            return False
        fx1, fy1, fx2, fy2 = face_box
        return not (x + w < fx1 or x > fx2 or y + h < fy1 or y > fy2)

    @staticmethod
    def _resize_rgba(image_rgba: np.ndarray, scale: float) -> np.ndarray:
        h, w = image_rgba.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(image_rgba, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _rotate_rgba(image_rgba: np.ndarray, angle: float) -> np.ndarray:
        h, w = image_rgba.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        return cv2.warpAffine(
            image_rgba,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    @staticmethod
    def _copy_to_with_alpha_mask(frame: np.ndarray, overlay_rgba: np.ndarray, x: int, y: int) -> None:
        frame_h, frame_w = frame.shape[:2]
        ov_h, ov_w = overlay_rgba.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + ov_w)
        y2 = min(frame_h, y + ov_h)
        if x1 >= x2 or y1 >= y2:
            return

        sx1 = x1 - x
        sy1 = y1 - y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        crop = overlay_rgba[sy1:sy2, sx1:sx2]
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)
        mask = crop[:, :, 3]
        roi = frame[y1:y2, x1:x2]
        cv2.copyTo(bgr, mask, roi)

    @staticmethod
    def _alpha_blend_rgba(
        frame: np.ndarray,
        overlay_rgba: np.ndarray,
        x: int,
        y: int,
        opacity: float,
    ) -> None:
        frame_h, frame_w = frame.shape[:2]
        ov_h, ov_w = overlay_rgba.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + ov_w)
        y2 = min(frame_h, y + ov_h)
        if x1 >= x2 or y1 >= y2:
            return

        sx1 = x1 - x
        sy1 = y1 - y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        crop = overlay_rgba[sy1:sy2, sx1:sx2]
        alpha = (crop[:, :, 3].astype(np.float32) / 255.0) * opacity
        if np.all(alpha <= 0.0):
            return

        alpha = alpha[:, :, None]
        overlay_bgr = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR).astype(np.float32)
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        blended = overlay_bgr * alpha + roi * (1.0 - alpha)
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    def _draw_save_notice(self, frame: np.ndarray) -> None:
        if self.save_notice_timer <= 0:
            return
        cv2.putText(
            frame,
            "★ SAVED!",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (0, 255, 100),
            2,
        )
        self.save_notice_timer -= 1

    @staticmethod
    def _ease_out(t: float) -> float:
        return 1.0 - (1.0 - t) ** 3

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t
