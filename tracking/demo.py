"""
MVLM Demo Script for Real-Time Video Tracking with Language Description

Usage:
    python tracking/demo.py --video <video_path> --text <description> [--config <yaml_name>] [--checkpoint <weight_path>] [--skip-selection]

Example:
    python tracking/demo.py --video test.mp4 --text "a red car" --config mvlm_b224_rgbn_xz_clip_full_cmloss_otu_3D-T-V-L_4gpu_b48_0.0003
"""

import os
import sys
import argparse
import subprocess
import threading
import queue
import time
import json
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2 as cv
import numpy as np
import torch

# Add project root to path
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker


class FFmpegCapture:
    """VideoCapture-compatible wrapper that uses ffmpeg subprocess for decoding.
    Used as a fallback when OpenCV cannot decode the video codec (e.g. AV1)."""

    def __init__(self, video_path):
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


_mjpeg_frame = None
_mjpeg_lock = threading.Lock()

# Control queues for web interface
_control_queue = queue.Queue()  # For pause, quit commands
_target_switch_queue = queue.Queue()
_current_text = None  # Will be set after args parsing
_is_paused = False  # For web UI to check pause state


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress per-request logs

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            html_path = os.path.join(os.path.dirname(__file__), 'demo_web_ui.html')
            with open(html_path, 'rb') as f:
                html = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html)
        elif self.path == '/stream':
            # MJPEG stream
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with _mjpeg_lock:
                        jpeg = _mjpeg_frame
                    if jpeg is not None:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == '/current_target':
            # Current target info API (also checks paused state from main loop)
            # Note: paused state is tracked in main() via a mutable list or we use a queue
            # For simplicity, we'll use a global for paused state
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            # Get paused state from global (set by main loop)
            paused = _is_paused if '_is_paused' in globals() else False
            self.wfile.write(json.dumps({'text': _current_text or '', 'paused': paused}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/new_target':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data)
                    _target_switch_queue.put(data)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status": "queued"}')
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "empty request"}')
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())

        elif self.path == '/control':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data)
                    _control_queue.put(data)
                    action = data.get('action', '')
                    messages = {'pause': 'Paused', 'resume': 'Resumed', 'quit': 'Quit signal sent'}
                    msg = messages.get(action, 'Done')
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'ok', 'message': msg}).encode())
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "empty request"}')
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _start_mjpeg_server(port):
    server = _ThreadedHTTPServer(('0.0.0.0', port), _MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"Web UI & Stream: http://<server-ip>:{port}  (open in browser to control tracking)")


def parse_args():
    parser = argparse.ArgumentParser(description='MVLM Demo - Real-time Video Tracking with Language')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--text', type=str, required=True, help='Natural language description of target object')
    parser.add_argument('--config', type=str,
                        default='mvlm_b224_rgbn_xz_clip_full_cmloss_otu_3D-T-V-L_4gpu_b48_0.0003',
                        help='YAML config name (without .yaml extension)')
    parser.add_argument('--checkpoint', type=str,
                        default='models/260126_sutrack_b224_rgbn_xz_clip_full_cmloss_otu_3D-T-V-L_4gpu_b48_0.0003_mean_ep0060.pth.tar',
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='TNL2K',
                        help='Dataset name for config (default: TNL2K for language-enabled tracking)')
    parser.add_argument('--skip-selection', action='store_true',
                        help='Skip interactive ROI selection and use center region of image')
    parser.add_argument('--bbox', type=str, default=None,
                        help='Specify bbox directly as "x,y,w,h" (overrides --skip-selection)')
    parser.add_argument('--half', action='store_true',
                        help='Use FP16 half-precision inference (faster on modern GPUs)')
    parser.add_argument('--frame_skip', type=int, default=0,
                        help='Skip N frames between each tracked frame (0=track every frame)')
    parser.add_argument('--no_display', action='store_true',
                        help='Disable imshow (measure pure inference speed without display overhead)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save tracked video to this path (e.g. result.mp4)')
    parser.add_argument('--stream_port', type=int, default=None,
                        help='Serve MJPEG stream on this port (e.g. 8080), viewable in browser')
    parser.add_argument('--device', type=str, default=None,
                        help='Force device: "cpu" or "cuda" (default: auto-detect)')
    return parser.parse_args()


def select_bbox_interactive(frame, text_prompt):
    """
    Display frame and let user select bounding box using mouse.

    Args:
        frame: First frame of video (BGR format)
        text_prompt: Natural language description to display

    Returns:
        bbox: [x, y, w, h] or None if cancelled
    """
    # Create a copy for display
    display_frame = frame.copy()

    # Add instruction text
    instruction = f"Target: {text_prompt}"
    cv.putText(display_frame, instruction, (10, 30),
              cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(display_frame, "Select ROI and press ENTER", (10, 65),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(display_frame, "Press ESC to cancel", (10, 95),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.putText(display_frame, "Press 's' to skip and use center region", (10, 125),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Select ROI
    roi = cv.selectROI("MVLM Demo - Select Target", display_frame, fromCenter=False, showCrosshair=True)

    cv.destroyWindow("MVLM Demo - Select Target")

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def get_default_bbox(frame):
    """
    Get a default bounding box at the center of the image.

    Args:
        frame: Input frame (BGR format)

    Returns:
        bbox: [x, y, w, h] - center region with half of image dimensions
    """
    h, w = frame.shape[:2]
    # Use center half of the image as default bbox
    bbox_w, bbox_h = w // 2, h // 2
    x = (w - bbox_w) // 2
    y = (h - bbox_h) // 2
    return [x, y, bbox_w, bbox_h]


def parse_bbox_string(bbox_str):
    """
    Parse bbox string in format "x,y,w,h".

    Args:
        bbox_str: String like "100,100,200,200"

    Returns:
        bbox: [x, y, w, h] or None if invalid
    """
    try:
        parts = bbox_str.split(',')
        if len(parts) != 4:
            return None
        x, y, w, h = [int(float(p)) for p in parts]
        if w <= 0 or h <= 0:
            return None
        return [x, y, w, h]
    except:
        return None


def draw_bbox(image, bbox, text_prompt, confidence=None):
    """
    Draw bounding box and text on image.

    Args:
        image: Image (BGR format)
        bbox: [x, y, w, h]
        text_prompt: Natural language description
        confidence: Optional confidence score

    Returns:
        Image with drawn bbox
    """
    img = image.copy()
    x, y, w, h = [int(v) for v in bbox]

    # Draw bounding box
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw text background
    text = f"Target: {text_prompt}"
    if confidence is not None:
        text += f" (conf: {confidence:.2f})"

    (text_w, text_h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), (0, 255, 0), -1)

    # Draw text
    cv.putText(img, text, (x + 5, y - 5),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return img


def _switch_target_interactive(tracker, frame):
    """Handle interactive target switching via imshow.

    Args:
        tracker: MVLM tracker instance
        frame: Current video frame (BGR format)

    Returns:
        bool: True if target was switched, False if cancelled
    """
    global _current_text

    print("\n" + "="*60)
    print("NEW TARGET SELECTION")
    print("="*60)

    # Use existing select_bbox_interactive function
    bbox = select_bbox_interactive(frame, "Select NEW target")

    if bbox is None:
        print("Cancelled. Resuming tracking...")
        return False

    print(f"Selected bbox: {bbox}")

    # Get new text from terminal
    new_text = input(f"Enter new description (current: {_current_text}): ").strip()

    if not new_text:
        print("No input. Resuming tracking...")
        return False

    # Re-initialize tracker with new bbox and text
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    init_info = {
        'init_bbox': bbox,
        'init_nlp': new_text,
        'seq_name': 'demo'
    }

    try:
        tracker.initialize(frame_rgb, init_info)
        _current_text = new_text
        print(f"Tracker re-initialized with: '{new_text}'")
        return True
    except Exception as e:
        print(f"Error re-initializing: {e}")
        print("Resuming with previous target...")
        return False


def main():
    global _current_text, _is_paused
    args = parse_args()
    _current_text = args.text  # Initialize global text description

    # Check if video file exists
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Check if checkpoint exists
    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    print("=" * 60)
    print("MVLM Demo - Real-time Video Tracking with Language")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Description: {args.text}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    if args.skip_selection:
        print("Bounding box: Auto (center region)")
    elif args.bbox:
        print(f"Bounding box: {args.bbox}")
    else:
        print("Bounding box: Interactive selection")
    print("=" * 60)

    # Initialize video capture
    cap = cv.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {args.video}")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("OpenCV cannot decode video; falling back to ffmpeg pipe decoder...")
        cap.release()
        cap = FFmpegCapture(args.video)
        fps = cap.get(cv.CAP_PROP_FPS)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read first frame from video (ffmpeg fallback also failed)")
            cap.release()
            sys.exit(1)

    # Get bounding box
    bbox = None

    if args.bbox:
        # Use provided bbox string
        bbox = parse_bbox_string(args.bbox)
        if bbox is None:
            print(f"Error: Invalid bbox format: {args.bbox}. Use format 'x,y,w,h'")
            cap.release()
            sys.exit(1)
        print(f"Using specified bbox: {bbox}")

    elif args.skip_selection:
        # Use default center bbox
        bbox = get_default_bbox(frame)
        print(f"Using default center bbox: {bbox}")

    else:
        # Interactive selection
        print("\nPlease select the target object in the window...")
        print("(Press ENTER to confirm, ESC to cancel, 's' to skip and use center)")
        bbox = select_bbox_interactive(frame, args.text)

        if bbox is None:
            print("Selection cancelled or skipped. Using default center bbox...")
            bbox = get_default_bbox(frame)

    print(f"Final bbox: {bbox}")

    # Initialize tracker
    print("\nInitializing tracker...")
    try:
        print("  [1/3] Loading checkpoint...")
        tracker_wrapper = Tracker(
            name='mvlm',
            parameter_name=args.config,
            exp_id='demo',
            dataset_name=args.dataset,
            weight_path=args.checkpoint
        )

        params = tracker_wrapper.get_parameters()
        params.debug = 0  # Disable debug mode for faster inference
        if args.device:
            params.device = args.device

        print("  [2/3] Creating tracker...")
        tracker = tracker_wrapper.create_tracker(params)

        if args.half:
            print("  FP16 autocast enabled (mixed precision)")

        # Initialize tracker with first frame
        print("  [3/3] Encoding text (CLIP) and initializing...")
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        init_info = {
            'init_bbox': bbox,
            'init_nlp': args.text,
            'seq_name': 'demo'
        }

        # NOTE: no autocast here - CLIP text encoder (initialize-only) requires float32
        tracker.initialize(frame_rgb, init_info)
        print("Tracker initialized successfully!")

    except Exception as e:
        print(f"Error initializing tracker: {e}")
        import traceback
        traceback.print_exc()
        cap.release()
        sys.exit(1)

    # Optional: MJPEG streaming server
    if args.stream_port:
        _start_mjpeg_server(args.stream_port)

    # Optional: video writer
    writer = None
    if args.output:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}")

    # Main tracking loop
    print("\nStarting tracking... Press ESC to quit, SPACE to pause, 'n' to switch target")
    if args.frame_skip > 0:
        print(f"Frame skip: {args.frame_skip} (tracking every {args.frame_skip + 1}th frame)")

    frame_count = 0
    paused = False
    display_frame = frame.copy()
    last_pred_bbox = bbox  # track last known bbox for text-only re-init

    # Background frame prefetch: read frames ahead of inference
    frame_queue = queue.Queue(maxsize=4)

    def _reader():
        skip = 0
        while True:
            ret, f = cap.read()
            if not ret:
                frame_queue.put((False, None))
                break
            if skip < args.frame_skip:
                skip += 1
                continue
            skip = 0
            frame_queue.put((True, f))

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    # FPS tracking
    fps_start = time.perf_counter()
    fps_count = 0
    measured_fps = 0.0
    t_display = 0.0

    if not args.no_display:
        cv.namedWindow("MVLM Demo", cv.WINDOW_AUTOSIZE)
    else:
        # Headless mode: accept keyboard commands from stdin
        def _stdin_keyboard_thread():
            print("[Console] Controls: p=pause  r=resume  q=quit  n=switch target")
            while True:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    key = line.strip().lower()
                    if key in ('q', 'quit'):
                        _control_queue.put({'action': 'quit'})
                    elif key in ('p', 'pause'):
                        _control_queue.put({'action': 'pause'})
                    elif key in ('r', 'resume'):
                        _control_queue.put({'action': 'resume'})
                    elif key in ('n',):
                        try:
                            new_text = input("  New description: ").strip()
                            if not new_text:
                                print("  Cancelled.")
                                continue
                            bbox_str = input("  Bbox x,y,w,h (Enter to skip): ").strip()
                            data = {'text': new_text}
                            if bbox_str:
                                parts = [int(v) for v in bbox_str.split(',')]
                                if len(parts) == 4:
                                    data['bbox'] = parts
                                else:
                                    print("  Invalid bbox, updating text only.")
                            _target_switch_queue.put(data)
                            print(f"  Queued target switch → '{new_text}'")
                        except Exception as e:
                            print(f"  Error: {e}")
                except Exception:
                    break
        threading.Thread(target=_stdin_keyboard_thread, daemon=True).start()

    while True:
        if not paused:
            ret, frame = frame_queue.get()
            if not ret:
                print("End of video reached")
                break

            frame_count += 1
            fps_count += 1

            # Run tracking
            try:
                t0 = time.perf_counter()
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                t1 = time.perf_counter()
                use_cuda = torch.cuda.is_available()
                with torch.amp.autocast('cuda', enabled=args.half and use_cuda):
                    results = tracker.track(frame_rgb)
                if use_cuda:
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                pred_bbox = results['target_bbox']
                last_pred_bbox = pred_bbox

                # Draw results
                display_frame = draw_bbox(frame, pred_bbox, _current_text)
                t3 = time.perf_counter()

                # Compute FPS every second
                elapsed = time.perf_counter() - fps_start
                if elapsed >= 1.0:
                    measured_fps = fps_count / elapsed
                    fps_count = 0
                    fps_start = time.perf_counter()
                    # print(f"[timing] cvt={1000*(t1-t0):.1f}ms  track={1000*(t2-t1):.1f}ms  draw={1000*(t3-t2):.1f}ms  display={1000*t_display:.1f}ms  FPS={measured_fps:.1f}")

                # Overlay stats
                cv.putText(display_frame, f"Frame: {frame_count}/{total_frames}",
                          (10, display_frame.shape[0] - 30),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv.putText(display_frame, f"FPS: {measured_fps:.1f}",
                          (10, display_frame.shape[0] - 10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            except Exception as e:
                print(f"Error during tracking: {e}")
                display_frame = frame
                cv.putText(display_frame, "Tracking Error!", (10, 30),
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Paused - keep showing last frame
            if not args.no_display:
                display_frame_paused = display_frame.copy()
                cv.putText(display_frame_paused, "PAUSED", (display_frame.shape[1]//2 - 50, 30),
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.imshow("MVLM Demo", display_frame_paused)
                key = cv.waitKey(30) & 0xFF
                if key == 27:
                    print("\nExiting...")
                    break
                elif key == 32:
                    paused = False
                    _is_paused = False
                    print("Resumed")
            else:
                time.sleep(0.03)

            # Must check control queue even while paused (otherwise Resume/Quit never fires)
            _paused_quit = False
            try:
                while not _control_queue.empty():
                    cmd = _control_queue.get_nowait()
                    action = cmd.get('action')
                    if action == 'resume':
                        paused = False
                        _is_paused = False
                        print("[Web] Resumed")
                    elif action == 'quit':
                        _paused_quit = True
                        print("\n[Web] Quit requested...")
            except queue.Empty:
                pass
            if _paused_quit:
                break
            continue

        # Save frame to video
        if writer is not None:
            writer.write(display_frame)

        # Push frame to MJPEG stream
        if args.stream_port:
            _, jpeg = cv.imencode('.jpg', display_frame, [cv.IMWRITE_JPEG_QUALITY, 80])
            with _mjpeg_lock:
                global _mjpeg_frame
                _mjpeg_frame = jpeg.tobytes()

        # Check for queued target switches (from web API)
        try:
            while not _target_switch_queue.empty():
                switch_data = _target_switch_queue.get_nowait()
                new_bbox = switch_data.get('bbox')
                new_text = switch_data.get('text', _current_text)
                _current_text = new_text

                init_bbox = new_bbox if new_bbox else last_pred_bbox
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                init_info = {'init_bbox': init_bbox, 'init_nlp': _current_text, 'seq_name': 'demo'}
                tracker.initialize(frame_rgb, init_info)
                last_pred_bbox = init_bbox
                src = f"bbox {init_bbox}" if new_bbox else "last tracked position"
                print(f"[API] Switched to: '{new_text}' ({src})")
        except queue.Empty:
            pass

        # Check for control commands (pause, quit) from web UI
        should_quit = False
        try:
            while not _control_queue.empty():
                cmd = _control_queue.get_nowait()
                action = cmd.get('action')

                if action == 'pause':
                    paused = True
                    _is_paused = True
                    print("[Web] Paused")
                elif action == 'resume':
                    paused = False
                    _is_paused = False
                    print("[Web] Resumed")
                elif action == 'quit':
                    should_quit = True
                    print("\n[Web] Quit requested...")
        except queue.Empty:
            pass

        if should_quit:
            break

        # Display
        t_disp0 = time.perf_counter()
        if not args.no_display:
            cv.imshow("MVLM Demo", display_frame)
            key = cv.waitKey(1) & 0xFF
        else:
            key = 0xFF
        t_display = time.perf_counter() - t_disp0

        if key == 27:  # ESC
            print("\nExiting...")
            break
        elif key == 32:  # SPACE
            paused = not paused
            _is_paused = paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord('n'):  # New target
            if args.no_display:
                print("[!] Target switch requires display (remove --no_display)")
                continue

            paused = True
            success = _switch_target_interactive(tracker, frame)
            if success:
                paused = False

    # Cleanup
    if writer is not None:
        writer.release()
    cap.release()
    cv.destroyAllWindows()
    print(f"\nTracked {frame_count} frames total")
    print("Demo finished!")


if __name__ == '__main__':
    main()
