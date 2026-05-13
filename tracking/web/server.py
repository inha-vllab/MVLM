"""
MVLM Tracker Service - Manages tracker lifecycle and state for the Web UI

This module provides a service class that wraps the MVLM tracker with
async-friendly methods and state management for the Web UI.
"""

import asyncio
import os
import sys
import threading
import time
from threading import Lock, Thread
from typing import Optional, Dict, Any, List
import queue

import cv2 as cv
import numpy as np
import torch

# Add project root to path
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker
from lib.config.mvlm.config import cfg, update_config_from_file


class FFmpegCapture:
    """VideoCapture-compatible wrapper using ffmpeg subprocess for decoding."""

    def __init__(self, video_path):
        import subprocess

        # Probe video properties with ffprobe
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        parts = probe.stdout.strip().split(',')
        self.width = int(parts[0]) if len(parts) > 0 else 0
        self.height = int(parts[1]) if len(parts) > 1 else 0
        num, den = (parts[2].split('/') if len(parts) > 2 else ('30', '1'))
        self.fps = float(num) / float(den)
        self.total_frames = int(parts[3]) if len(parts) > 3 and parts[3].strip() else -1

        self._proc = subprocess.Popen(
            ['ffmpeg', '-i', video_path, '-f', 'rawvideo', '-pix_fmt', 'bgr24',
             '-loglevel', 'error', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._frame_size = self.width * self.height * 3

    def get(self, prop_id):
        if prop_id == cv.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        if prop_id == cv.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop_id == cv.CAP_PROP_FRAME_HEIGHT:
            return self.height
        return 0

    def read(self):
        raw = self._proc.stdout.read(self._frame_size)
        if len(raw) < self._frame_size:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def release(self):
        self._proc.stdout.close()
        self._proc.terminate()
        self._proc.wait()


class MVLMTrackerService:
    """
    Manages MVLM tracker lifecycle and state for the Web UI.

    This class provides:
    - Async-friendly tracker initialization and control
    - Video source management (file, webcam)
    - Target switching with language descriptions
    - Real-time status and frame access
    - Parameter management
    """

    def __init__(self):
        # Tracker state
        self.tracker = None
        self.tracker_wrapper = None
        self.params = None
        self.is_running = False
        self.is_paused = False
        self.is_initialized = False

        # Video state
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30.0
        self.video_width = 0
        self.video_height = 0

        # Target state
        self.current_text = ""
        self.current_bbox = None
        self.last_pred_bbox = None

        # First frame for preview / bbox selection
        self._first_frame = None

        # Threading
        self._frame_lock = Lock()
        self._state_lock = Lock()
        self._tracking_thread = None
        self._latest_frame = None
        self._stop_event = threading.Event()

        # Queue for target switches
        self._target_queue = queue.Queue()

        # Performance tracking
        self._fps_start = time.perf_counter()
        self._fps_count = 0
        self.measured_fps = 0.0

        # Config
        self.current_config = None
        self.current_checkpoint = None
        self.half_precision = False

    def get_status(self) -> Dict[str, Any]:
        """Get current tracker status."""
        with self._state_lock:
            return {
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "is_initialized": self.is_initialized,
                "model_loaded": self.tracker_wrapper is not None,
                "video_loaded": self.cap is not None,
                "current_text": self.current_text,
                "current_frame": self.current_frame,
                "total_frames": self.total_frames,
                "fps": self.measured_fps,
                "video_width": self.video_width,
                "video_height": self.video_height,
                "config": self.current_config,
                "checkpoint": self.current_checkpoint,
            }

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest processed frame (thread-safe)."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    async def load_model(
        self,
        config_name: str,
        checkpoint_path: str,
        dataset_name: str = "TNL2K",
        half_precision: bool = False
    ):
        """Load model checkpoint and initialize tracker wrapper."""
        with self._state_lock:
            # Check if checkpoint exists
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            print(f"Loading model: {config_name} from {checkpoint_path}")

            try:
                self.tracker_wrapper = Tracker(
                    name='mvlm',
                    parameter_name=config_name,
                    exp_id='webui',
                    dataset_name=dataset_name,
                    weight_path=checkpoint_path
                )

                self.params = self.tracker_wrapper.get_parameters()
                self.params.debug = 0

                # Pass device override if set via CLI --device
                device_override = os.environ.get('MVLM_DEVICE')
                if device_override:
                    self.params.device = device_override

                self.current_config = config_name
                self.current_checkpoint = checkpoint_path
                self.half_precision = half_precision

                print("Model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")

    async def load_video(self, video_path: str):
        """Load video file for tracking."""
        with self._state_lock:
            if self.is_running:
                raise RuntimeError("Cannot load video while tracking is running")

            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")

            self.video_path = video_path

            # Try OpenCV first, fallback to ffmpeg
            cap = cv.VideoCapture(video_path)
            if not cap.isOpened():
                cap = FFmpegCapture(video_path)

            self.fps = cap.get(cv.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            self.video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            # Read first frame
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError("Failed to read first frame from video")

            self.cap = cap
            self._first_frame = frame.copy()
            self.current_frame = 0

            print(f"Video loaded: {self.video_width}x{self.video_height}, "
                  f"{self.fps:.2f} FPS, {self.total_frames} frames")

    async def load_video_from_url(self, url: str):
        """Load video from URL."""
        # Simple implementation using OpenCV
        with self._state_lock:
            if self.is_running:
                raise RuntimeError("Cannot load video while tracking is running")

            cap = cv.VideoCapture(url)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video from URL: {url}")

            self.fps = cap.get(cv.CAP_PROP_FPS) or 30.0
            self.total_frames = -1  # Unknown for streams
            self.video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            self.cap = cap
            self.video_path = url
            self.current_frame = 0

    async def load_webcam(self, device_id: int = 0):
        """Load webcam for tracking."""
        with self._state_lock:
            if self.is_running:
                raise RuntimeError("Cannot load webcam while tracking is running")

            cap = cv.VideoCapture(device_id)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open webcam device {device_id}")

            self.fps = cap.get(cv.CAP_PROP_FPS) or 30.0
            self.total_frames = -1
            self.video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            self.cap = cap
            self.video_path = f"webcam://{device_id}"
            self.current_frame = 0

    async def start_tracking(
        self,
        text: str,
        bbox: Optional[List[int]] = None
    ):
        """Start tracking with the given target description."""
        with self._state_lock:
            if self.is_running:
                raise RuntimeError("Tracking already running")

            if not self.tracker_wrapper:
                raise RuntimeError("No model loaded")

            if not self.cap:
                raise RuntimeError("No video source loaded")

            # Create tracker instance
            self.tracker = self.tracker_wrapper.create_tracker(self.params)

            # Use stored first frame (avoids skipping frame 0)
            if self._first_frame is not None:
                frame = self._first_frame.copy()
                # Rewind video to frame 0 so tracking loop starts from frame 1
                try:
                    self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                except Exception:
                    pass  # FFmpegCapture may not support seeking
            else:
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("Failed to read first frame")

            # Determine bbox
            if bbox is None:
                # Use center region as default
                h, w = frame.shape[:2]
                bbox = [w // 4, h // 4, w // 2, h // 2]

            # Initialize tracker
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            init_info = {
                'init_bbox': bbox,
                'init_nlp': text,
                'seq_name': 'webui'
            }

            self.tracker.initialize(frame_rgb, init_info)

            self.current_text = text
            self.current_bbox = bbox
            self.last_pred_bbox = bbox
            self.is_initialized = True
            self.is_running = True
            self.is_paused = False
            self._stop_event.clear()

            # Start tracking thread
            self._tracking_thread = Thread(target=self._tracking_loop, daemon=True)
            self._tracking_thread.start()

    async def switch_target(
        self,
        text: str,
        bbox: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Switch to a new target."""
        with self._state_lock:
            if not self.is_initialized:
                raise RuntimeError("Tracker not initialized")

            # Queue the target switch
            self._target_queue.put({
                'text': text,
                'bbox': bbox
            })

            self.current_text = text

            return {
                "text": text,
                "bbox": bbox,
                "source": "last position" if bbox is None else "specified bbox"
            }

    def pause(self):
        """Pause tracking."""
        with self._state_lock:
            self.is_paused = True

    def resume(self):
        """Resume tracking."""
        with self._state_lock:
            self.is_paused = False

    async def stop(self):
        """Stop tracking and cleanup resources."""
        with self._state_lock:
            self.is_running = False
            self._stop_event.set()

        # Wait for tracking thread to finish
        if self._tracking_thread and self._tracking_thread.is_alive():
            self._tracking_thread.join(timeout=2.0)

        await self.cleanup()

    async def cleanup(self):
        """Cleanup resources."""
        with self._state_lock:
            if self.cap:
                self.cap.release()
                self.cap = None

            self.is_running = False
            self.is_initialized = False
            self._latest_frame = None

    async def seek_frame(self, frame_number: int):
        """Seek to a specific frame (for videos only)."""
        with self._state_lock:
            if not self.cap or self.video_path is None:
                raise RuntimeError("No video loaded")

            if self.video_path.startswith("webcam://"):
                raise RuntimeError("Cannot seek in webcam stream")

            # For file-based videos, we need to re-open and seek
            was_running = self.is_running
            if was_running:
                await self.stop()

            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number

    def update_parameters(self, parameters: Dict[str, Any]):
        """Update tracker parameters."""
        with self._state_lock:
            if self.params:
                for key, value in parameters.items():
                    if hasattr(self.params, key):
                        setattr(self.params, key, value)

    def _tracking_loop(self):
        """Main tracking loop (runs in separate thread)."""
        fps_start = time.perf_counter()
        fps_count = 0

        while not self._stop_event.is_set():
            with self._state_lock:
                if self.is_paused or not self.is_running:
                    time.sleep(0.03)
                    continue

            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("End of video stream")
                break

            self.current_frame += 1
            fps_count += 1

            # Run tracking
            try:
                t0 = time.perf_counter()
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                use_cuda = torch.cuda.is_available()
                with torch.amp.autocast('cuda', enabled=self.half_precision and use_cuda):
                    results = self.tracker.track(frame_rgb)

                if use_cuda:
                    torch.cuda.synchronize()
                pred_bbox = results['target_bbox']
                self.last_pred_bbox = pred_bbox

                # Check for target switches
                try:
                    while not self._target_queue.empty():
                        switch_data = self._target_queue.get_nowait()
                        new_text = switch_data.get('text', self.current_text)
                        new_bbox = switch_data.get('bbox')

                        with self._state_lock:
                            self.current_text = new_text

                        init_bbox = new_bbox if new_bbox else self.last_pred_bbox
                        init_info = {
                            'init_bbox': init_bbox,
                            'init_nlp': new_text,
                            'seq_name': 'webui'
                        }
                        self.tracker.initialize(frame_rgb, init_info)
                        print(f"Switched target: '{new_text}'")
                except queue.Empty:
                    pass

                # Draw results
                display_frame = self._draw_bbox(frame, pred_bbox, self.current_text)

                # Update FPS
                elapsed = time.perf_counter() - fps_start
                if elapsed >= 1.0:
                    self.measured_fps = fps_count / elapsed
                    fps_count = 0
                    fps_start = time.perf_counter()

                # Store latest frame
                with self._frame_lock:
                    self._latest_frame = display_frame

            except Exception as e:
                print(f"Tracking error: {e}")
                with self._frame_lock:
                    self._latest_frame = frame
                    cv.putText(self._latest_frame, "Tracking Error!",
                              (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def _draw_bbox(self, image: np.ndarray, bbox: List, text_prompt: str) -> np.ndarray:
        """Draw bounding box and text on image."""
        img = image.copy()
        x, y, w, h = [int(v) for v in bbox]

        # Draw bounding box
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw text background
        text = f"Target: {text_prompt[:40]}"
        (text_w, text_h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), (0, 255, 0), -1)

        # Draw text
        cv.putText(img, text, (x + 5, y - 5),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return img

