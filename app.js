import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

const WINDOW_NAME = "DAB DETECTOR 9000";
const BOTTOM_TEXT = "[Bottom Text]";

const REQUIRED_LANDMARKS = [
  "LEFT_WRIST",
  "LEFT_ELBOW",
  "LEFT_SHOULDER",
  "RIGHT_WRIST",
  "RIGHT_ELBOW",
  "RIGHT_SHOULDER",
  "NOSE",
];

const OPTIONAL_LANDMARKS = ["LEFT_EYE", "RIGHT_EYE"];

const POSE_LANDMARK_INDEX = {
  NOSE: 0,
  LEFT_EYE: 2,
  RIGHT_EYE: 5,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
};

const SPRITE_FILES = [
  "assets/bear.png",
  "assets/doritos.png",
  "assets/illuminati.png",
  "assets/mtndew.png",
  "assets/sonic.png",
];

class BrowserAudioEngine {
  constructor(audioPath = "assets/Timeline 1.mp3") {
    this.audio = new Audio(audioPath);
    this.audio.loop = true;
    this.audio.preload = "auto";
    this.audioEnabled = true;
  }

  async unlock() {
    if (!this.audioEnabled) {
      return;
    }

    try {
      this.audio.volume = 0;
      await this.audio.play();
      this.audio.pause();
      this.audio.currentTime = 0;
      this.audio.volume = 1;
    } catch (error) {
      this.audioEnabled = false;
      console.warn("[audio] unlock failed", error);
    }
  }

  async play() {
    if (!this.audioEnabled) {
      return;
    }

    try {
      await this.audio.play();
    } catch (error) {
      console.warn("[audio] play failed", error);
    }
  }

  stop() {
    this.audio.pause();
    this.audio.currentTime = 0;
  }
}

class Exporter {
  constructor() {
    this.saved = false;
  }

  reset() {
    this.saved = false;
  }

  maybeExport(canvas, dabFrameCount) {
    if (dabFrameCount !== 10 || this.saved) {
      return null;
    }

    const filename = `dab_${formatTimestamp(new Date())}.jpg`;
    canvas.toBlob((blob) => {
      if (!blob) {
        return;
      }

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, "image/jpeg", 0.92);

    this.saved = true;
    console.info(`[export] Saved -> ${filename}`);
    return filename;
  }
}

class PoseEngine {
  constructor() {
    this.landmarker = null;
    this.frameCounter = 0;
  }

  async init(modelPath = "assets/pose_landmarker_heavy.task") {
    if (this.landmarker) {
      return;
    }

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm",
    );

    try {
      this.landmarker = await this.createLandmarker(vision, modelPath, "GPU");
    } catch (error) {
      console.warn("[pose] GPU init failed, falling back to CPU", error);
      this.landmarker = await this.createLandmarker(vision, modelPath);
    }

    console.info("[pose] Backend: mediapipe tasks PoseLandmarker");
  }

  createLandmarker(vision, modelPath, delegate = null) {
    const baseOptions = { modelAssetPath: modelPath };
    if (delegate) {
      baseOptions.delegate = delegate;
    }

    return PoseLandmarker.createFromOptions(vision, {
      baseOptions,
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.6,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputSegmentationMasks: false,
    });
  }

  classify(video, timestampMs, frameWidth, frameHeight) {
    if (!this.landmarker) {
      return emptyPoseResult();
    }

    this.frameCounter += 1;

    let capturedResult = null;
    const maybeResult = this.landmarker.detectForVideo(
      video,
      timestampMs,
      (result) => {
        capturedResult = result;
      },
    );
    const result = capturedResult ?? maybeResult;
    const landmarks = result?.landmarks?.[0];
    if (!landmarks) {
      return emptyPoseResult();
    }

    const pixelLandmarks = this.toPixelLandmarks(
      landmarks,
      frameWidth,
      frameHeight,
    );
    const nosePx = pixelLandmarks.NOSE ?? null;

    if (!this.visibilityGuardPassed(landmarks)) {
      return {
        isDab: false,
        landmarks: pixelLandmarks,
        extendedSide: null,
        nosePx,
      };
    }

    const leftDiag = this.checkOrientation(landmarks, "LEFT");
    const rightDiag = this.checkOrientation(landmarks, "RIGHT");
    return this.buildPoseResultFromOrientationChecks(
      leftDiag.ok,
      rightDiag.ok,
      leftDiag.angle,
      rightDiag.angle,
      pixelLandmarks,
      nosePx,
    );
  }

  buildPoseResultFromOrientationChecks(
    leftOk,
    rightOk,
    leftAngle,
    rightAngle,
    pixelLandmarks,
    nosePx,
  ) {
    const isDab = leftOk || rightOk;
    if (!isDab) {
      return {
        isDab: false,
        landmarks: pixelLandmarks,
        extendedSide: null,
        nosePx,
      };
    }

    let extendedSide = "RIGHT";
    if (leftOk && rightOk) {
      extendedSide = leftAngle >= rightAngle ? "LEFT" : "RIGHT";
    } else if (leftOk) {
      extendedSide = "LEFT";
    }

    return {
      isDab: true,
      landmarks: pixelLandmarks,
      extendedSide,
      nosePx,
    };
  }

  toPixelLandmarks(landmarks, frameWidth, frameHeight) {
    const output = {};
    for (const name of [...REQUIRED_LANDMARKS, ...OPTIONAL_LANDMARKS]) {
      const landmark = landmarks[POSE_LANDMARK_INDEX[name]];
      if (!landmark) {
        continue;
      }
      output[name] = landmarkToMirroredPx(
        landmark.x,
        landmark.y,
        frameWidth,
        frameHeight,
      );
    }
    return output;
  }

  visibilityGuardPassed(landmarks) {
    return REQUIRED_LANDMARKS.every((name) => {
      const landmark = landmarks[POSE_LANDMARK_INDEX[name]];
      return effectiveVisibility(landmark) > 0.5;
    });
  }

  checkOrientation(landmarks, extendedSide) {
    const shoulder = landmarks[POSE_LANDMARK_INDEX[`${extendedSide}_SHOULDER`]];
    const elbow = landmarks[POSE_LANDMARK_INDEX[`${extendedSide}_ELBOW`]];
    const wrist = landmarks[POSE_LANDMARK_INDEX[`${extendedSide}_WRIST`]];

    const elbowAngle = elbowAngleDegrees(
      [shoulder.x, shoulder.y],
      [elbow.x, elbow.y],
      [wrist.x, wrist.y],
    );
    const straight = elbowAngle > 155.0;
    const wristRaised = wrist.y < shoulder.y;

    return {
      ok: straight && wristRaised,
      angle: elbowAngle,
      straight,
      wristRaised,
    };
  }
}

class FXEngine {
  constructor() {
    this.sunglassesImage = null;
    this.spriteAssets = [];
    this.activeSprites = [];
    this.rainbowBase = null;
    this.hueOffset = 0;
    this.saveNoticeTimer = 0;
  }

  async loadAssets() {
    const [sunglasses, sprites] = await Promise.all([
      loadImage("assets/sunglasses.png").catch(() => createFallbackSunglasses()),
      Promise.all(SPRITE_FILES.map((path) => loadImage(path).catch(() => null))),
    ]);

    this.sunglassesImage = sunglasses;
    this.spriteAssets = sprites.filter(Boolean);
  }

  reset() {
    this.activeSprites = [];
    this.rainbowBase = null;
    this.hueOffset = 0;
    this.saveNoticeTimer = 0;
  }

  notifySaved() {
    this.saveNoticeTimer = 60;
  }

  apply(ctx, frameWidth, frameHeight, poseResult, dabFrameCount) {
    if (dabFrameCount === 1) {
      this.onDabActivated(frameWidth, frameHeight, poseResult);
    }

    if (
      !this.rainbowBase ||
      this.rainbowBase.width !== frameWidth ||
      this.rainbowBase.height !== frameHeight
    ) {
      this.buildRainbow(frameWidth, frameHeight);
    }

    this.applyRainbow(ctx, frameWidth);
    this.updateAndDrawSprites(ctx, frameWidth, frameHeight, poseResult, dabFrameCount);
    this.applySunglasses(ctx, poseResult, dabFrameCount);
    this.drawSaveNotice(ctx, frameHeight);
  }

  onDabActivated(frameWidth, frameHeight, poseResult) {
    this.hueOffset = 0;
    this.buildRainbow(frameWidth, frameHeight);
    this.activeSprites = [];

    const spawnCount = randomInt(3, 5);
    for (let index = 0; index < spawnCount; index += 1) {
      this.spawnSprite(frameWidth, frameHeight, poseResult);
    }
  }

  buildRainbow(frameWidth, frameHeight) {
    const rainbowCanvas = document.createElement("canvas");
    rainbowCanvas.width = frameWidth;
    rainbowCanvas.height = frameHeight;
    const rainbowCtx = rainbowCanvas.getContext("2d");
    const gradient = rainbowCtx.createLinearGradient(0, 0, frameWidth, 0);

    gradient.addColorStop(0.0, "#ff003c");
    gradient.addColorStop(0.16, "#ff8a00");
    gradient.addColorStop(0.33, "#ffe600");
    gradient.addColorStop(0.5, "#00d084");
    gradient.addColorStop(0.66, "#00c2ff");
    gradient.addColorStop(0.83, "#4b4bff");
    gradient.addColorStop(1.0, "#ff00d4");

    rainbowCtx.fillStyle = gradient;
    rainbowCtx.fillRect(0, 0, frameWidth, frameHeight);
    this.rainbowBase = rainbowCanvas;
  }

  applyRainbow(ctx, frameWidth) {
    if (!this.rainbowBase) {
      return;
    }

    this.hueOffset = (this.hueOffset + 5) % Math.max(1, frameWidth);
    ctx.save();
    ctx.globalAlpha = 0.4;
    ctx.drawImage(this.rainbowBase, -this.hueOffset, 0);
    ctx.drawImage(this.rainbowBase, frameWidth - this.hueOffset, 0);
    ctx.restore();
  }

  applySunglasses(ctx, poseResult, dabFrameCount) {
    if (!this.sunglassesImage) {
      return;
    }

    const leftEye = poseResult.landmarks.LEFT_EYE;
    const rightEye = poseResult.landmarks.RIGHT_EYE;
    if (!leftEye || !rightEye) {
      return;
    }

    const interEyeDistance = distance(leftEye, rightEye);
    if (interEyeDistance < 2) {
      return;
    }

    const sourceWidth = getAssetWidth(this.sunglassesImage);
    const sourceHeight = getAssetHeight(this.sunglassesImage);
    const targetWidth = Math.max(1, Math.round(interEyeDistance * 2.2));
    const targetHeight = Math.max(
      1,
      Math.round(targetWidth * (sourceHeight / sourceWidth)),
    );

    const angle = Math.atan2(
      rightEye[1] - leftEye[1],
      rightEye[0] - leftEye[0],
    );
    const eyeCenterX = Math.round((leftEye[0] + rightEye[0]) * 0.5);
    const eyeCenterY = Math.round((leftEye[1] + rightEye[1]) * 0.5);
    const animationT = Math.min(dabFrameCount / 5.0, 1.0);
    const eased = easeOut(animationT);
    const drawY = lerp(eyeCenterY - 150, eyeCenterY, eased);

    ctx.save();
    ctx.translate(eyeCenterX, drawY);
    ctx.rotate(angle);
    ctx.drawImage(
      this.sunglassesImage,
      -targetWidth / 2,
      -targetHeight / 2,
      targetWidth,
      targetHeight,
    );
    ctx.restore();
  }

  updateAndDrawSprites(ctx, frameWidth, frameHeight, poseResult, dabFrameCount) {
    if (this.spriteAssets.length > 0 && this.activeSprites.length < 10) {
      let spawnCount = 0;
      if (dabFrameCount <= 12 && dabFrameCount % 3 === 0) {
        spawnCount += 1;
      } else if (dabFrameCount % 12 === 0) {
        spawnCount += 1;
      }
      if (Math.random() < 0.1) {
        spawnCount += 1;
      }

      for (let index = 0; index < spawnCount; index += 1) {
        this.spawnSprite(frameWidth, frameHeight, poseResult);
      }
    }

    const survivors = [];
    const centerBox = this.centerAvoidBox(frameWidth, frameHeight);

    for (const sprite of this.activeSprites) {
      sprite.age += 1;
      sprite.x += sprite.vx;
      sprite.y += sprite.vy;
      sprite.angle = (sprite.angle + sprite.angularVelocity) % 360.0;

      const fadeIn = Math.min(1.0, sprite.age / 4.0);
      const fadeOut = Math.min(1.0, Math.max(0, sprite.maxAge - sprite.age) / 10.0);
      const opacity = fadeIn * fadeOut;
      if (opacity <= 0.0 || sprite.age >= sprite.maxAge) {
        continue;
      }

      const pulse =
        1.0 +
        Math.sin(sprite.age * sprite.pulseSpeed + sprite.wobblePhase) *
          sprite.pulseAmount;
      const scale = Math.max(0.06, sprite.baseScale * pulse);
      const scaledWidth = Math.max(1, Math.round(getAssetWidth(sprite.image) * scale));
      const scaledHeight = Math.max(1, Math.round(getAssetHeight(sprite.image) * scale));
      const rotatedSize = rotatedBounds(
        scaledWidth,
        scaledHeight,
        degreesToRadians(sprite.angle),
      );

      const wobbleT = sprite.age * sprite.wobbleSpeed + sprite.wobblePhase;
      let drawX = Math.round(
        sprite.x + Math.cos(wobbleT) * sprite.wobbleAmount,
      );
      let drawY = Math.round(
        sprite.y + Math.sin(wobbleT * 1.3) * sprite.wobbleAmount,
      );

      [drawX, drawY] = this.deflectSpriteFromBox(
        sprite,
        drawX,
        drawY,
        rotatedSize.width,
        rotatedSize.height,
        centerBox,
      );

      if (
        isFarOffscreen(
          frameWidth,
          frameHeight,
          drawX,
          drawY,
          rotatedSize.width,
          rotatedSize.height,
          180,
        )
      ) {
        continue;
      }

      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.translate(drawX + rotatedSize.width / 2, drawY + rotatedSize.height / 2);
      ctx.rotate(degreesToRadians(sprite.angle));
      ctx.drawImage(
        sprite.image,
        -scaledWidth / 2,
        -scaledHeight / 2,
        scaledWidth,
        scaledHeight,
      );
      ctx.restore();

      survivors.push(sprite);
    }

    this.activeSprites = survivors;
  }

  spawnSprite(frameWidth, frameHeight, poseResult) {
    if (this.spriteAssets.length === 0) {
      return;
    }

    const spriteImage = pickRandom(this.spriteAssets);
    const spriteHeight = getAssetHeight(spriteImage);
    const spriteWidth = getAssetWidth(spriteImage);
    const baseScale = this.randomSpriteScale(
      frameWidth,
      frameHeight,
      spriteWidth,
      spriteHeight,
    );
    const scaledWidth = Math.max(1, Math.round(spriteWidth * baseScale));
    const scaledHeight = Math.max(1, Math.round(spriteHeight * baseScale));
    const centerBox = this.centerAvoidBox(frameWidth, frameHeight);

    let faceBox = null;
    if (poseResult.nosePx) {
      const [nx, ny] = poseResult.nosePx;
      faceBox = [nx - 100, ny - 100, nx + 100, ny + 100];
    }

    if (this.activeSprites.length >= 10) {
      this.activeSprites.shift();
    }

    let motion;
    if (Math.random() < 0.7) {
      motion = this.spawnDashMotion(
        frameWidth,
        frameHeight,
        scaledWidth,
        scaledHeight,
        centerBox,
      );
    } else {
      const avoidBoxes = [faceBox, centerBox].filter(Boolean);
      motion = this.spawnDriftMotion(
        frameWidth,
        frameHeight,
        scaledWidth,
        scaledHeight,
        avoidBoxes,
      );
    }

    this.activeSprites.push({
      image: spriteImage,
      x: motion.x,
      y: motion.y,
      vx: motion.vx,
      vy: motion.vy,
      baseScale,
      pulseAmount: randomFloat(0.08, 0.28),
      pulseSpeed: randomFloat(0.34, 0.82),
      age: 0,
      maxAge: randomInt(22, 46),
      angle: randomFloat(0.0, 360.0),
      angularVelocity: pickRandom([-1.0, 1.0]) * randomFloat(14.0, 42.0),
      wobbleAmount: randomFloat(4.0, 18.0),
      wobbleSpeed: randomFloat(0.3, 0.78),
      wobblePhase: randomFloat(0.0, Math.PI * 2),
    });
  }

  randomSpriteScale(frameWidth, frameHeight, spriteWidth, spriteHeight) {
    const frameMin = Math.min(frameWidth, frameHeight);
    const spriteLongest = Math.max(spriteWidth, spriteHeight);
    const targetLongest = randomFloat(frameMin * 0.16, frameMin * 0.3);
    const scale = targetLongest / Math.max(1, spriteLongest);
    return clamp(scale, 0.12, 0.58);
  }

  spawnDashMotion(frameWidth, frameHeight, spriteWidth, spriteHeight, centerBox) {
    const frameMin = Math.min(frameWidth, frameHeight);
    const speed = randomFloat(
      Math.max(11.0, frameMin * 0.03),
      Math.max(18.0, frameMin * 0.085),
    );
    const sway = randomFloat(-speed * 0.55, speed * 0.55);
    const margin = Math.max(spriteWidth, spriteHeight) + 30.0;
    const edge = pickRandom(["left", "right", "top", "bottom"]);
    const [cx1, cy1, cx2, cy2] = centerBox;

    if (edge === "left") {
      return {
        x: -spriteWidth - randomFloat(0.0, margin),
        y: chooseOuterCoordinate(
          -spriteHeight * 0.25,
          cy1 - spriteHeight - 12.0,
          cy2 + 12.0,
          frameHeight - spriteHeight * 0.75,
          0.0,
          Math.max(0.0, frameHeight - spriteHeight),
        ),
        vx: speed,
        vy: sway,
      };
    }

    if (edge === "right") {
      return {
        x: frameWidth + randomFloat(0.0, margin),
        y: chooseOuterCoordinate(
          -spriteHeight * 0.25,
          cy1 - spriteHeight - 12.0,
          cy2 + 12.0,
          frameHeight - spriteHeight * 0.75,
          0.0,
          Math.max(0.0, frameHeight - spriteHeight),
        ),
        vx: -speed,
        vy: sway,
      };
    }

    if (edge === "top") {
      return {
        x: chooseOuterCoordinate(
          -spriteWidth * 0.25,
          cx1 - spriteWidth - 12.0,
          cx2 + 12.0,
          frameWidth - spriteWidth * 0.75,
          0.0,
          Math.max(0.0, frameWidth - spriteWidth),
        ),
        y: -spriteHeight - randomFloat(0.0, margin),
        vx: sway,
        vy: speed,
      };
    }

    return {
      x: chooseOuterCoordinate(
        -spriteWidth * 0.25,
        cx1 - spriteWidth - 12.0,
        cx2 + 12.0,
        frameWidth - spriteWidth * 0.75,
        0.0,
        Math.max(0.0, frameWidth - spriteWidth),
      ),
      y: frameHeight + randomFloat(0.0, margin),
      vx: sway,
      vy: -speed,
    };
  }

  spawnDriftMotion(frameWidth, frameHeight, spriteWidth, spriteHeight, avoidBoxes) {
    const maxX = Math.max(0, frameWidth - spriteWidth);
    const maxY = Math.max(0, frameHeight - spriteHeight);

    let x = 0.0;
    let y = 0.0;
    for (let index = 0; index < 30; index += 1) {
      x = maxX > 0 ? randomFloat(0.0, maxX) : 0.0;
      y = maxY > 0 ? randomFloat(0.0, maxY) : 0.0;
      if (!intersectsAnyBox(x, y, spriteWidth, spriteHeight, avoidBoxes)) {
        break;
      }
    }

    const speed = randomFloat(4.0, Math.max(8.0, Math.min(frameWidth, frameHeight) * 0.04));
    const direction = randomFloat(0.0, Math.PI * 2);
    return {
      x,
      y,
      vx: Math.cos(direction) * speed,
      vy: Math.sin(direction) * speed,
    };
  }

  centerAvoidBox(frameWidth, frameHeight) {
    const boxWidth = Math.round(frameWidth * 0.34);
    const boxHeight = Math.round(frameHeight * 0.42);
    const cx = Math.floor(frameWidth / 2);
    const cy = Math.floor(frameHeight / 2);
    return [
      Math.max(0, cx - Math.floor(boxWidth / 2)),
      Math.max(0, cy - Math.floor(boxHeight / 2)),
      Math.min(frameWidth, cx + Math.floor(boxWidth / 2)),
      Math.min(frameHeight, cy + Math.floor(boxHeight / 2)),
    ];
  }

  deflectSpriteFromBox(sprite, drawX, drawY, spriteWidth, spriteHeight, avoidBox) {
    if (!intersectsFaceBox(drawX, drawY, spriteWidth, spriteHeight, avoidBox)) {
      return [drawX, drawY];
    }

    const [x1, y1, x2, y2] = avoidBox;
    const spriteCenterX = drawX + spriteWidth / 2;
    const spriteCenterY = drawY + spriteHeight / 2;
    const boxCenterX = (x1 + x2) / 2;
    const boxCenterY = (y1 + y2) / 2;
    const margin = 14.0;

    if (Math.abs(spriteCenterX - boxCenterX) >= Math.abs(spriteCenterY - boxCenterY)) {
      if (spriteCenterX < boxCenterX) {
        sprite.x = x1 - spriteWidth - margin;
        sprite.vx = -Math.abs(sprite.vx) - 1.5;
      } else {
        sprite.x = x2 + margin;
        sprite.vx = Math.abs(sprite.vx) + 1.5;
      }
      drawX = Math.round(sprite.x);
    } else {
      if (spriteCenterY < boxCenterY) {
        sprite.y = y1 - spriteHeight - margin;
        sprite.vy = -Math.abs(sprite.vy) - 1.5;
      } else {
        sprite.y = y2 + margin;
        sprite.vy = Math.abs(sprite.vy) + 1.5;
      }
      drawY = Math.round(sprite.y);
    }

    return [drawX, drawY];
  }

  drawSaveNotice(ctx, frameHeight) {
    if (this.saveNoticeTimer <= 0) {
      return;
    }

    ctx.save();
    ctx.fillStyle = "rgb(0, 255, 100)";
    ctx.font = '700 40px "Trebuchet MS", Verdana, sans-serif';
    ctx.textBaseline = "alphabetic";
    ctx.fillText("* SAVED!", 20, frameHeight - 20);
    ctx.restore();

    this.saveNoticeTimer -= 1;
  }
}

class BrowserDabApp {
  constructor() {
    this.video = document.getElementById("camera");
    this.stageCanvas = document.getElementById("stage");
    this.stageCtx = this.stageCanvas.getContext("2d");
    this.launchOverlay = document.getElementById("launchOverlay");
    this.startButton = document.getElementById("startButton");
    this.launchStatus = document.getElementById("launchStatus");

    this.renderCanvas = document.createElement("canvas");
    this.renderCtx = this.renderCanvas.getContext("2d");

    this.pose = new PoseEngine();
    this.fx = new FXEngine();
    this.audio = new BrowserAudioEngine();
    this.exporter = new Exporter();

    this.state = "IDLE";
    this.holdCounter = 0;
    this.dabCounter = 0;
    this.lastVideoTime = -1;
    this.lastPoseResult = emptyPoseResult();
    this.running = false;
    this.stream = null;

    this.startButton.addEventListener("click", () => {
      this.start().catch((error) => {
        console.error(error);
        this.setStatus(error.message || "Unable to start the browser app.");
        this.startButton.disabled = false;
      });
    });

    window.addEventListener("resize", () => this.resizeStage());
    window.addEventListener("beforeunload", () => this.stop());
  }

  async start() {
    this.startButton.disabled = true;
    this.setStatus("Loading the browser version...");
    document.title = WINDOW_NAME;

    await enterFullscreenIfPossible(document.documentElement);
    await Promise.all([
      this.pose.init(),
      this.fx.loadAssets(),
      this.audio.unlock(),
      this.startCamera(),
    ]);

    this.launchOverlay.classList.add("is-hidden");
    this.running = true;
    this.setStatus("");
    this.resizeStage();
    window.requestAnimationFrame((timestampMs) => this.renderFrame(timestampMs));
  }

  async startCamera() {
    const constraints = {
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: window.innerWidth || 1280 },
        height: { ideal: window.innerHeight || 720 },
      },
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.stream;

    await new Promise((resolve) => {
      this.video.onloadedmetadata = () => resolve();
    });

    await this.video.play();
  }

  stop() {
    this.running = false;
    this.audio.stop();
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
  }

  renderFrame(timestampMs) {
    if (!this.running) {
      return;
    }

    const frameWidth = this.video.videoWidth;
    const frameHeight = this.video.videoHeight;
    if (frameWidth <= 0 || frameHeight <= 0) {
      window.requestAnimationFrame((nextTimestampMs) => this.renderFrame(nextTimestampMs));
      return;
    }

    if (
      this.renderCanvas.width !== frameWidth ||
      this.renderCanvas.height !== frameHeight
    ) {
      this.renderCanvas.width = frameWidth;
      this.renderCanvas.height = frameHeight;
    }

    this.drawMirroredVideoFrame(frameWidth, frameHeight);

    let poseResult = emptyPoseResult();
    if (this.video.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = this.video.currentTime;
      poseResult = this.pose.classify(
        this.video,
        timestampMs,
        frameWidth,
        frameHeight,
      );
      this.lastPoseResult = poseResult;
    } else {
      poseResult = this.lastPoseResult;
    }

    if (this.state === "IDLE") {
      if (poseResult.isDab) {
        this.holdCounter += 1;
        if (this.holdCounter >= 2) {
          this.state = "DAB";
          this.dabCounter = 0;
          this.exporter.reset();
          this.audio.play();
        }
      } else {
        this.holdCounter = 0;
      }
    } else if (this.state === "DAB") {
      this.dabCounter += 1;
      this.fx.apply(
        this.renderCtx,
        frameWidth,
        frameHeight,
        poseResult,
        this.dabCounter,
      );

      const savedName = this.exporter.maybeExport(this.renderCanvas, this.dabCounter);
      if (savedName) {
        this.fx.notifySaved();
      }

      if (!poseResult.isDab) {
        this.state = "IDLE";
        this.holdCounter = 0;
        this.dabCounter = 0;
        this.audio.stop();
        this.fx.reset();
      }
    }

    this.drawBottomText();
    this.drawStage();

    window.requestAnimationFrame((nextTimestampMs) => this.renderFrame(nextTimestampMs));
  }

  drawMirroredVideoFrame(frameWidth, frameHeight) {
    this.renderCtx.save();
    this.renderCtx.setTransform(1, 0, 0, 1, 0, 0);
    this.renderCtx.clearRect(0, 0, frameWidth, frameHeight);
    this.renderCtx.translate(frameWidth, 0);
    this.renderCtx.scale(-1, 1);
    this.renderCtx.drawImage(this.video, 0, 0, frameWidth, frameHeight);
    this.renderCtx.restore();
  }

  drawBottomText() {
    const frameWidth = this.renderCanvas.width;
    const frameHeight = this.renderCanvas.height;
    const fontSize = Math.max(24, Math.floor(frameWidth / 12));
    const strokeWidth = Math.max(2, Math.floor(fontSize / 18));

    this.renderCtx.save();
    this.renderCtx.font = `700 ${fontSize}px Impact, Haettenschweiler, "Arial Narrow Bold", sans-serif`;
    this.renderCtx.textAlign = "center";
    this.renderCtx.textBaseline = "bottom";
    this.renderCtx.lineJoin = "round";
    this.renderCtx.lineWidth = strokeWidth * 2;
    this.renderCtx.strokeStyle = "rgb(0, 0, 0)";
    this.renderCtx.fillStyle = "rgb(255, 255, 255)";

    const textX = frameWidth / 2;
    const textY = Math.max(fontSize, frameHeight - 50);
    this.renderCtx.strokeText(BOTTOM_TEXT, textX, textY);
    this.renderCtx.fillText(BOTTOM_TEXT, textX, textY);
    this.renderCtx.restore();
  }

  resizeStage() {
    const width = Math.max(1, window.innerWidth);
    const height = Math.max(1, window.innerHeight);
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const targetWidth = Math.round(width * dpr);
    const targetHeight = Math.round(height * dpr);

    if (
      this.stageCanvas.width !== targetWidth ||
      this.stageCanvas.height !== targetHeight
    ) {
      this.stageCanvas.width = targetWidth;
      this.stageCanvas.height = targetHeight;
      this.stageCanvas.style.width = `${width}px`;
      this.stageCanvas.style.height = `${height}px`;
    }
  }

  drawStage() {
    const viewportWidth = Math.max(1, window.innerWidth);
    const viewportHeight = Math.max(1, window.innerHeight);
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    this.stageCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.stageCtx.clearRect(0, 0, viewportWidth, viewportHeight);

    const drawRect = coverRect(
      this.renderCanvas.width,
      this.renderCanvas.height,
      viewportWidth,
      viewportHeight,
    );
    this.stageCtx.drawImage(
      this.renderCanvas,
      drawRect.x,
      drawRect.y,
      drawRect.width,
      drawRect.height,
    );
  }

  setStatus(message) {
    this.launchStatus.textContent = message;
  }
}

function emptyPoseResult() {
  return {
    isDab: false,
    landmarks: {},
    extendedSide: null,
    nosePx: null,
  };
}

function effectiveVisibility(landmark) {
  if (!landmark) {
    return 0.0;
  }
  if (typeof landmark.visibility === "number") {
    return landmark.visibility;
  }
  if (typeof landmark.presence === "number") {
    return landmark.presence;
  }
  return 1.0;
}

function landmarkToMirroredPx(x, y, frameWidth, frameHeight) {
  return [
    clamp(Math.round((1 - x) * frameWidth), 0, frameWidth - 1),
    clamp(Math.round(y * frameHeight), 0, frameHeight - 1),
  ];
}

function elbowAngleDegrees(shoulder, elbow, wrist) {
  const vecA = [shoulder[0] - elbow[0], shoulder[1] - elbow[1]];
  const vecB = [wrist[0] - elbow[0], wrist[1] - elbow[1]];

  const normA = Math.hypot(vecA[0], vecA[1]);
  const normB = Math.hypot(vecB[0], vecB[1]);
  if (normA === 0 || normB === 0) {
    return 0.0;
  }

  const cosTheta = clamp(
    (vecA[0] * vecB[0] + vecA[1] * vecB[1]) / (normA * normB),
    -1.0,
    1.0,
  );
  return radiansToDegrees(Math.acos(cosTheta));
}

function coverRect(sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const scale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const width = Math.max(targetWidth, Math.round(sourceWidth * scale));
  const height = Math.max(targetHeight, Math.round(sourceHeight * scale));
  return {
    x: Math.round((targetWidth - width) / 2),
    y: Math.round((targetHeight - height) / 2),
    width,
    height,
  };
}

function isFarOffscreen(frameWidth, frameHeight, x, y, width, height, padding) {
  return (
    x + width < -padding ||
    x > frameWidth + padding ||
    y + height < -padding ||
    y > frameHeight + padding
  );
}

function intersectsAnyBox(x, y, width, height, boxes) {
  return boxes.some((box) => intersectsFaceBox(x, y, width, height, box));
}

function intersectsFaceBox(x, y, width, height, faceBox) {
  if (!faceBox) {
    return false;
  }

  const [fx1, fy1, fx2, fy2] = faceBox;
  return !(x + width < fx1 || x > fx2 || y + height < fy1 || y > fy2);
}

function chooseOuterCoordinate(
  lowerStart,
  lowerEnd,
  upperStart,
  upperEnd,
  fallbackStart,
  fallbackEnd,
) {
  const options = [];
  if (lowerEnd > lowerStart) {
    options.push([lowerStart, lowerEnd]);
  }
  if (upperEnd > upperStart) {
    options.push([upperStart, upperEnd]);
  }

  if (options.length === 0) {
    return fallbackEnd > fallbackStart
      ? randomFloat(fallbackStart, fallbackEnd)
      : fallbackStart;
  }

  const [start, end] = pickRandom(options);
  return randomFloat(start, end);
}

function rotatedBounds(width, height, angleRadians) {
  const cos = Math.abs(Math.cos(angleRadians));
  const sin = Math.abs(Math.sin(angleRadians));
  return {
    width: Math.max(1, Math.round(height * sin + width * cos)),
    height: Math.max(1, Math.round(height * cos + width * sin)),
  };
}

function createFallbackSunglasses() {
  const canvas = document.createElement("canvas");
  canvas.width = 220;
  canvas.height = 60;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#000";
  ctx.beginPath();
  ctx.ellipse(54, 30, 44, 22, 0, 0, Math.PI * 2);
  ctx.ellipse(166, 30, 44, 22, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillRect(98, 27, 24, 6);

  ctx.fillStyle = "#fff";
  ctx.fillRect(40, 18, 4, 4);
  ctx.fillRect(152, 18, 4, 4);
  return canvas;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load ${src}`));
    image.src = src;
  });
}

function getAssetWidth(asset) {
  return asset.naturalWidth || asset.videoWidth || asset.width;
}

function getAssetHeight(asset) {
  return asset.naturalHeight || asset.videoHeight || asset.height;
}

function formatTimestamp(date) {
  const pad = (value) => String(value).padStart(2, "0");
  return [
    `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}`,
    `${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`,
  ].join("_");
}

function easeOut(t) {
  return 1.0 - (1.0 - t) ** 3;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function distance(pointA, pointB) {
  return Math.hypot(pointB[0] - pointA[0], pointB[1] - pointA[1]);
}

function randomFloat(min, max) {
  return min + Math.random() * (max - min);
}

function randomInt(min, max) {
  return Math.floor(randomFloat(min, max + 1));
}

function pickRandom(values) {
  return values[Math.floor(Math.random() * values.length)];
}

function degreesToRadians(value) {
  return (value * Math.PI) / 180;
}

function radiansToDegrees(value) {
  return (value * 180) / Math.PI;
}

async function enterFullscreenIfPossible(element) {
  if (!document.fullscreenEnabled || typeof element.requestFullscreen !== "function") {
    return;
  }

  try {
    if (!document.fullscreenElement) {
      await element.requestFullscreen();
    }
  } catch (error) {
    console.warn("[ui] fullscreen request failed", error);
  }
}

const app = new BrowserDabApp();
window.dabDetectorApp = app;
