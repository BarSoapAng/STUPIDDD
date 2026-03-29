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
    x: float
    y: float
    vx: float
    vy: float
    base_scale: float
    pulse_amount: float
    pulse_speed: float
    age: int
    max_age: int
    angle: float
    angular_velocity: float
    wobble_amount: float
    wobble_speed: float
    wobble_phase: float


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
        self.hue_offset = (self.hue_offset + 5) % 180
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
        anim_t = min(dab_frame_count / 5.0, 1.0)
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
        if self.sprite_assets and len(self.active_sprites) < 10:
            spawn_count = 0
            if dab_frame_count <= 12 and dab_frame_count % 3 == 0:
                spawn_count += 1
            elif dab_frame_count % 12 == 0:
                spawn_count += 1
            if random.random() < 0.10:
                spawn_count += 1
            for _ in range(spawn_count):
                self._spawn_sprite(frame.shape[:2], pose_result)

        survivors: list[ActiveSprite] = []
        center_box = self._center_avoid_box(frame.shape[:2])
        for sprite in self.active_sprites:
            sprite.age += 1
            sprite.x += sprite.vx
            sprite.y += sprite.vy
            sprite.angle = (sprite.angle + sprite.angular_velocity) % 360.0

            fade_in = min(1.0, sprite.age / 4.0)
            fade_out = min(1.0, max(0, sprite.max_age - sprite.age) / 10.0)
            opacity = fade_in * fade_out
            if opacity <= 0.0 or sprite.age >= sprite.max_age:
                continue

            pulse = 1.0 + math.sin(sprite.age * sprite.pulse_speed + sprite.wobble_phase) * sprite.pulse_amount
            scale = max(0.06, sprite.base_scale * pulse)
            scaled = self._resize_rgba(sprite.image, scale)
            rotated = self._rotate_rgba(scaled, sprite.angle)

            wobble_t = sprite.age * sprite.wobble_speed + sprite.wobble_phase
            draw_x = int(round(sprite.x + math.cos(wobble_t) * sprite.wobble_amount))
            draw_y = int(round(sprite.y + math.sin(wobble_t * 1.3) * sprite.wobble_amount))
            draw_x, draw_y = self._deflect_sprite_from_box(
                sprite,
                draw_x,
                draw_y,
                rotated.shape[1],
                rotated.shape[0],
                center_box,
            )

            if self._is_far_offscreen(
                frame.shape[:2],
                draw_x,
                draw_y,
                rotated.shape[1],
                rotated.shape[0],
                padding=180,
            ):
                continue

            self._alpha_blend_rgba(frame, rotated, draw_x, draw_y, opacity)
            survivors.append(sprite)

        self.active_sprites = survivors

    def _spawn_sprite(self, frame_shape_hw: tuple[int, int], pose_result: PoseResult) -> None:
        if not self.sprite_assets:
            return

        sprite_image = random.choice(self.sprite_assets)
        sprite_h, sprite_w = sprite_image.shape[:2]
        base_scale = self._random_sprite_scale(frame_shape_hw, (sprite_h, sprite_w))
        scaled_h, scaled_w = self._scaled_size((sprite_h, sprite_w), base_scale)
        center_box = self._center_avoid_box(frame_shape_hw)

        face_box = None
        if pose_result.nose_px is not None:
            nx, ny = pose_result.nose_px
            face_box = (nx - 100, ny - 100, nx + 100, ny + 100)

        if len(self.active_sprites) >= 10:
            self.active_sprites.pop(0)

        if random.random() < 0.7:
            x, y, vx, vy = self._spawn_dash_motion(frame_shape_hw, (scaled_h, scaled_w), center_box)
        else:
            x, y, vx, vy = self._spawn_drift_motion(
                frame_shape_hw,
                (scaled_h, scaled_w),
                [box for box in (face_box, center_box) if box is not None],
            )

        self.active_sprites.append(
            ActiveSprite(
                image=sprite_image,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                base_scale=base_scale,
                pulse_amount=random.uniform(0.08, 0.28),
                pulse_speed=random.uniform(0.34, 0.82),
                age=0,
                max_age=random.randint(22, 46),
                angle=random.uniform(0.0, 360.0),
                angular_velocity=random.choice((-1.0, 1.0)) * random.uniform(14.0, 42.0),
                wobble_amount=random.uniform(4.0, 18.0),
                wobble_speed=random.uniform(0.30, 0.78),
                wobble_phase=random.uniform(0.0, math.tau),
            )
        )

    @staticmethod
    def _random_sprite_scale(
        frame_shape_hw: tuple[int, int],
        sprite_shape_hw: tuple[int, int],
    ) -> float:
        frame_min = min(frame_shape_hw)
        sprite_longest = max(sprite_shape_hw)
        target_longest = random.uniform(frame_min * 0.16, frame_min * 0.30)
        scale = target_longest / max(1, sprite_longest)
        return min(0.58, max(0.12, scale))

    @staticmethod
    def _scaled_size(shape_hw: tuple[int, int], scale: float) -> tuple[int, int]:
        h, w = shape_hw
        return max(1, int(h * scale)), max(1, int(w * scale))

    @staticmethod
    def _spawn_dash_motion(
        frame_shape_hw: tuple[int, int],
        sprite_shape_hw: tuple[int, int],
        center_box: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float]:
        frame_h, frame_w = frame_shape_hw
        sprite_h, sprite_w = sprite_shape_hw
        frame_min = min(frame_shape_hw)
        speed = random.uniform(max(11.0, frame_min * 0.03), max(18.0, frame_min * 0.085))
        sway = random.uniform(-speed * 0.55, speed * 0.55)
        margin = max(sprite_h, sprite_w) + 30.0
        edge = random.choice(("left", "right", "top", "bottom"))
        cx1, cy1, cx2, cy2 = center_box

        if edge == "left":
            x = -sprite_w - random.uniform(0.0, margin)
            y = FXEngine._choose_outer_coordinate(
                -sprite_h * 0.25,
                cy1 - sprite_h - 12.0,
                cy2 + 12.0,
                frame_h - sprite_h * 0.75,
                0.0,
                max(0.0, frame_h - sprite_h),
            )
            return x, y, speed, sway
        if edge == "right":
            x = frame_w + random.uniform(0.0, margin)
            y = FXEngine._choose_outer_coordinate(
                -sprite_h * 0.25,
                cy1 - sprite_h - 12.0,
                cy2 + 12.0,
                frame_h - sprite_h * 0.75,
                0.0,
                max(0.0, frame_h - sprite_h),
            )
            return x, y, -speed, sway
        if edge == "top":
            x = FXEngine._choose_outer_coordinate(
                -sprite_w * 0.25,
                cx1 - sprite_w - 12.0,
                cx2 + 12.0,
                frame_w - sprite_w * 0.75,
                0.0,
                max(0.0, frame_w - sprite_w),
            )
            y = -sprite_h - random.uniform(0.0, margin)
            return x, y, sway, speed
        x = FXEngine._choose_outer_coordinate(
            -sprite_w * 0.25,
            cx1 - sprite_w - 12.0,
            cx2 + 12.0,
            frame_w - sprite_w * 0.75,
            0.0,
            max(0.0, frame_w - sprite_w),
        )
        y = frame_h + random.uniform(0.0, margin)
        return x, y, sway, -speed

    @staticmethod
    def _spawn_drift_motion(
        frame_shape_hw: tuple[int, int],
        sprite_shape_hw: tuple[int, int],
        avoid_boxes: list[tuple[int, int, int, int]],
    ) -> tuple[float, float, float, float]:
        frame_h, frame_w = frame_shape_hw
        sprite_h, sprite_w = sprite_shape_hw
        max_x = max(0, frame_w - sprite_w)
        max_y = max(0, frame_h - sprite_h)

        x = 0.0
        y = 0.0
        for _ in range(30):
            x = random.uniform(0.0, max_x) if max_x > 0 else 0.0
            y = random.uniform(0.0, max_y) if max_y > 0 else 0.0
            if not FXEngine._intersects_any_box(int(x), int(y), sprite_w, sprite_h, avoid_boxes):
                break

        speed = random.uniform(4.0, max(8.0, min(frame_shape_hw) * 0.04))
        direction = random.uniform(0.0, math.tau)
        vx = math.cos(direction) * speed
        vy = math.sin(direction) * speed
        return x, y, vx, vy

    @staticmethod
    def _center_avoid_box(frame_shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape_hw
        box_w = int(frame_w * 0.34)
        box_h = int(frame_h * 0.42)
        cx = frame_w // 2
        cy = frame_h // 2
        return (
            max(0, cx - box_w // 2),
            max(0, cy - box_h // 2),
            min(frame_w, cx + box_w // 2),
            min(frame_h, cy + box_h // 2),
        )

    @staticmethod
    def _choose_outer_coordinate(
        lower_start: float,
        lower_end: float,
        upper_start: float,
        upper_end: float,
        fallback_start: float,
        fallback_end: float,
    ) -> float:
        options: list[tuple[float, float]] = []
        if lower_end > lower_start:
            options.append((lower_start, lower_end))
        if upper_end > upper_start:
            options.append((upper_start, upper_end))
        if not options:
            return random.uniform(fallback_start, fallback_end) if fallback_end > fallback_start else fallback_start
        start, end = random.choice(options)
        return random.uniform(start, end)

    @staticmethod
    def _deflect_sprite_from_box(
        sprite: ActiveSprite,
        draw_x: int,
        draw_y: int,
        sprite_w: int,
        sprite_h: int,
        avoid_box: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        if not FXEngine._intersects_face_box(draw_x, draw_y, sprite_w, sprite_h, avoid_box):
            return draw_x, draw_y

        x1, y1, x2, y2 = avoid_box
        sprite_cx = draw_x + (sprite_w / 2.0)
        sprite_cy = draw_y + (sprite_h / 2.0)
        box_cx = (x1 + x2) / 2.0
        box_cy = (y1 + y2) / 2.0
        margin = 14.0

        if abs(sprite_cx - box_cx) >= abs(sprite_cy - box_cy):
            if sprite_cx < box_cx:
                sprite.x = float(x1 - sprite_w - margin)
                sprite.vx = -abs(sprite.vx) - 1.5
            else:
                sprite.x = float(x2 + margin)
                sprite.vx = abs(sprite.vx) + 1.5
            draw_x = int(round(sprite.x))
        else:
            if sprite_cy < box_cy:
                sprite.y = float(y1 - sprite_h - margin)
                sprite.vy = -abs(sprite.vy) - 1.5
            else:
                sprite.y = float(y2 + margin)
                sprite.vy = abs(sprite.vy) + 1.5
            draw_y = int(round(sprite.y))

        return draw_x, draw_y

    @staticmethod
    def _intersects_any_box(
        x: int,
        y: int,
        w: int,
        h: int,
        boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        return any(FXEngine._intersects_face_box(x, y, w, h, box) for box in boxes)

    @staticmethod
    def _is_far_offscreen(
        frame_shape_hw: tuple[int, int],
        x: int,
        y: int,
        w: int,
        h: int,
        padding: int,
    ) -> bool:
        frame_h, frame_w = frame_shape_hw
        return x + w < -padding or x > frame_w + padding or y + h < -padding or y > frame_h + padding

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
