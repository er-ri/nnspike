#!/usr/bin/env python3

import argparse
from email.mime import base
import sys
import time

from nnspike.unit import ETRobot, KeyboardController

import cv2
import numpy as np
from nnspike.constants import BASE_SPEED, HIGH_SPEED_BASE, CAMERA_WIDTH, CAMERA_HEIGHT, OFFSET_Y, ROI_CNN, Mode, ROI_COLOR
from nnspike.utils import PIDController, SensorRecorder, draw_driving_info, get_line_edges_at_y, find_bottle_center, find_blue_target_center, get_virtual_line_target_x, get_offset_pixels
import threading
import queue

class Video:
    def __init__(self):
        """
        常にrealtimeモードのみ。fps/buffer_sizeは内部定数で管理
        """
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        buffer_size = 1
        self.cap = cv2.VideoCapture(0)  # USBカメラ前提で0固定
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    # set_mode廃止（モード切替不可）

    def warmup(self, count=10):
        """
        カメラウォームアップ用: 指定回数read()を呼ぶ
        """
        for _ in range(count):
            self.read()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        """
        最新フレームのみ返す（スレッドで取得した最新フレーム）
        """
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.running = False
        self.thread.join()
        # self.cap.release()

def handle_status_and_video(frame, status, mode, left_speed, right_speed,
                           record_sensor_data, sensor_recorder, save_camera_video, video_writer):
    """status取得・センサー記録・動画送信処理"""
    if record_sensor_data and sensor_recorder is not None:
        sensor_recorder.log_frame_data(status, mode, left_speed, right_speed)
    if save_camera_video and video_writer is not None:
        video_writer.write(frame)
    
    return True
    
# StateFlagsクラス（バックアップより）
class StateFlags:

    def __init__(self):
        self._first_key_used = False
        self._force_sensor_mode_switch_enabled = False

    @property
    def first_key_used(self):
        return self._first_key_used
    @first_key_used.setter
    def first_key_used(self, value: bool):
        self._first_key_used = value

    @property
    def force_sensor_mode_switch_enabled(self):
        return self._force_sensor_mode_switch_enabled
    @force_sensor_mode_switch_enabled.setter
    def force_sensor_mode_switch_enabled(self, value: bool):
        self._force_sensor_mode_switch_enabled = value

    def enable_force_sensor_mode_switch(self):
        self.force_sensor_mode_switch_enabled = True
    def disable_force_sensor_mode_switch(self):
        self.force_sensor_mode_switch_enabled = False

def wait_for_start(et, keyboard, state_flags, manual_mode=False):
    """
    起動時にフォースセンサーの接続状態を確認し、
    フォースセンサーまたは有効なモードキーでロボットをスタートさせる。
    終了時は first_key（最初に押されたキーまたはforceセンサー）を返す。
    """
    force_baseline = None
    try:
        status_init = et.get_spike_status()
        force_baseline = getattr(status_init.sensors, "force", None)
        if force_baseline is None:
            time.sleep(1)
            status_init = et.get_spike_status()
            force_baseline = getattr(status_init.sensors, "force", None)
        if force_baseline is not None:
            print(f"Force sensor is active. Baseline: {force_baseline}. Press to start.")
            print("\r", end="")
            sys.stdout.flush()
        else:
            print("Force sensor is NOT detected. Please check connection.")
            print("\r", end="")
            sys.stdout.flush()
    except Exception:
        print("Force sensor check failed. Please check hardware.")

    print("Press the force sensor (change >100) or any mode key to start...")
    started = False
    first_key = None
    while not started and keyboard.running:
        # 1秒間で最大50回リトライ
        for _ in range(50):
            status = et.get_spike_status()
            force_val = getattr(status.sensors, "force", None)
            key = keyboard.get_key()
            # forceセンサー押下でスタート（初期値から絶対値100以上変動）
            if (force_val is not None and force_baseline is not None and abs(force_val - force_baseline) >= 100):
                print(f"Start! (force changed: {force_baseline} -> {force_val})")
                started = True
                first_key = "__force__"  # forceセンサーでスタートした場合はダミー値をセット
                break
            # manual_mode時のみ有効なモードキーでスタート
            elif manual_mode and key is not None and keyboard.is_mode_key(key):
                print("Start!")
                first_key = key
                started = True
                break
            if not keyboard.running:
                print("Quitting before start. Exiting...")
                et.stop()
                keyboard.cleanup()
                return None
            time.sleep(0.02)
    # スタート決定後に一度だけモード切替有効化/無効化を判定
    if started:
        if first_key == "__force__":
            state_flags.enable_force_sensor_mode_switch()
        else:
            state_flags.disable_force_sensor_mode_switch()

    return first_key

def main(record_sensor_data=False, save_camera_video=False, course="right", course_type="upper", manual_mode=False):

    def handle_debug_output(loop_start, loop_end, debug_state, min_interval=0.017):
        debug_state['counter'] += 1
        # --- min_interval引数で周期調整＋デバッグ出力 ---
        dt = loop_end - loop_start
        sleep_sec = min_interval - dt if dt < min_interval else 0
        sleep_ms = sleep_sec * 1000
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        total_ms = (time.time() - loop_start) * 1000
        # print(f"[DEBUG] dt={dt*1000:.2f}ms, sleep={sleep_ms:.2f}ms, total={total_ms:.2f}ms")

    state_flags = StateFlags()
    # Generate timestamp for consistent naming if recording is enabled
    TIMESTAMP = time.strftime("%Y%m%d%H%M%S", time.localtime()) if (record_sensor_data or save_camera_video) else None

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    record_queue = None
    record_thread = None
    record_thread_running = False
    if record_sensor_data:
        sensor_recorder = SensorRecorder(timestamp=TIMESTAMP)
        sensor_recorder.start_recording()
        record_queue = queue.Queue(maxsize=200)
        record_thread_running = True
        def record_worker():
            while record_thread_running or not record_queue.empty():
                try:
                    args = record_queue.get(timeout=0.1)
                    if args is not None:
                        status, mode, left_speed, right_speed = args
                        sensor_recorder.log_frame_data(status, mode, left_speed, right_speed)
                except queue.Empty:
                    continue
        record_thread = threading.Thread(target=record_worker, daemon=True)
        record_thread.start()

    video_writer = None
    video_filename = None
    video_queue = None
    video_thread = None
    video_thread_running = False
    if save_camera_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
        video_filename = f"storage/videos/{TIMESTAMP}_picamera.avi"
        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=30,
            frameSize=(CAMERA_WIDTH, CAMERA_HEIGHT),
        )
        video_queue = queue.Queue(maxsize=100)
        video_thread_running = True
        def video_save_worker():
            while video_thread_running or not video_queue.empty():
                try:
                    frame = video_queue.get(timeout=0.1)
                    if frame is not None:
                        video_writer.write(frame)
                except queue.Empty:
                    continue
        video_thread = threading.Thread(target=video_save_worker, daemon=True)
        video_thread.start()

    # Initialize edge following preference based on the course parameter
    et = ETRobot()

    # Initialize robot, keyboard controller
    keyboard = KeyboardController()

    et.set_motor_relative_position(left_positon=0, right_position=0)

    # --- Videoクラスでカメラ起動・ウォームアップ ---
    video = Video()
    video.warmup()
    # カメラの実際のFPS値を表示
    actual_fps = video.cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Camera actual FPS: {actual_fps}")
    # --- スタート待ち ---
    first_key = wait_for_start(et, keyboard, state_flags, manual_mode=manual_mode)
    if first_key is None:
        # videoを使用している場合は解放
        video.release()
        return

    # wait_for_start()の後にmodeの初期値を決定
    if state_flags.force_sensor_mode_switch_enabled:
        mode = Mode.TEST
    else:
        mode = Mode.PAUSE

    # --- 変数初期化 ---
    left_speed = None
    right_speed = None
    last_frame_time = None
    last_frame_data = None
    turn_left_started = False
    turn_right_started = False
    yaw_start_left = None
    yaw_start_right = None

    # 毎回判定する必要のないフラグを事前計算
    need_status = (record_sensor_data and sensor_recorder is not None)

    # debug状態を辞書で管理（エレガントな状態管理）
    debug_state = {
        'counter': 0,
        'last_print': time.time()
    }
    target_yaw = None
    prev_mode_test = False
    try:
        while et.is_running:
            loop_start = time.time()
            # --- カメラフレーム取得・保存処理を復活 ---
            ret, frame = video.read()
            loop_camera = time.time()
            # 物理的なフレーム周期計測用
            if last_frame_time is None:
                last_frame_time = loop_camera
            dt_camera_physical = loop_camera - last_frame_time
            # フレームが変化した場合のみ周期を出力
            if hasattr(video, 'frame') and video.frame is not None:
                if last_frame_data is None or not np.array_equal(video.frame, last_frame_data):
                    print(f"[CAMERA_PHYSICAL] dt={dt_camera_physical*1000:.2f}ms")
                    last_frame_time = loop_camera
                    last_frame_data = video.frame.copy() if isinstance(video.frame, np.ndarray) else video.frame
            # retがFalseでもframeがNoneでなければ前回画像で制御継続
            if not ret and (frame is None):
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # retがFalseかつframeがNoneでなければ、前回画像で制御継続（警告のみ）
            if not ret and (frame is not None):
                print("[WARN] Camera frame not updated, using previous frame.")
            if save_camera_video and video_writer is not None and frame is not None and isinstance(frame, np.ndarray):
                try:
                    video_queue.put_nowait(frame)
                except queue.Full:
                    pass  # キューが満杯なら捨てる

            # status取得・センサー記録（別スレッド化）
            if need_status:
                status = et.get_spike_status()
                try:
                    record_queue.put_nowait((status, mode, left_speed, right_speed))
                except queue.Full:
                    pass  # キューが満杯なら捨てる

            # キー処理とモード切替（統合版）
            if manual_mode and not state_flags.first_key_used:
                key = first_key
                state_flags.first_key_used = True
            else:
                key = keyboard.get_key()
                if not manual_mode and not state_flags.first_key_used:
                    state_flags.first_key_used = True
            
            result = keyboard.get_mode_from_key(key)
            if result is not None:
                mode_result, msg = result
            else:
                mode_result, msg = None, None
            if mode_result == "quit":
                print(msg)
                keyboard.running = False
                break
            elif manual_mode and mode_result is not None:
                mode = mode_result
                print(msg)

            # モード分岐
            if mode == Mode.FORWARD:
                # シンプルに直進のみ
                left_speed = BASE_SPEED
                right_speed = BASE_SPEED
                et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
            elif mode == Mode.BACKWARD:
                # set_motor_backward_speedで後退（左右は入れ替えない）
                left_speed = BASE_SPEED
                right_speed = BASE_SPEED
                et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
            elif mode == Mode.TURN_LEFT:
                if not turn_left_started:
                    et.set_start_yaw()
                    turn_left_started = True
                left_speed, right_speed = et.yaw_turn_control(
                    side="left",
                    threshold_deg=90.0,
                    base_speed=BASE_SPEED
                )
                et.set_motor_speed(left_speed=left_speed, right_speed=right_speed)
                print(f"[TURN_LEFT] yaw={et.get_yaw():.2f}, yaw_start={et.get_start_yaw():.2f}, diff={et.get_yaw() - et.get_start_yaw():.2f}")
                if left_speed == 0 and right_speed == 0:
                    print(f"[TURN_LEFT] reached -90 deg and stopped | yaw={et.get_yaw():.2f}")
                    et.set_motor_forward_speed(left_speed=0, right_speed=0)
                    mode = Mode.PAUSE
                    turn_left_started = False
            elif mode == Mode.TURN_RIGHT:
                if not turn_right_started:
                    et.set_start_yaw()
                    turn_right_started = True
                left_speed, right_speed = et.yaw_turn_control(
                    side="right",
                    threshold_deg=90.0,
                    base_speed=BASE_SPEED
                )
                et.set_motor_speed(left_speed=left_speed, right_speed=right_speed)
                print(f"[TURN_RIGHT] yaw={et.get_yaw():.2f}, yaw_start={et.get_start_yaw():.2f}, diff={et.get_yaw() - et.get_start_yaw():.2f}")
                if left_speed == 0 and right_speed == 0:
                    print(f"[TURN_RIGHT] reached +90 deg and stopped | yaw={et.get_yaw():.2f}")
                    et.set_motor_forward_speed(left_speed=0, right_speed=0)
                    mode = Mode.PAUSE
                    turn_right_started = False
            elif mode == Mode.TEST:
                if not prev_mode_test:
                    et.set_start_yaw()
                prev_mode_test = True
                left_speed, right_speed = et.yaw_straight_control(
                    base_speed=HIGH_SPEED_BASE,
                    kp=1.0,
                    deadband=2.0
                )
                print(f"[TEST DEBUG] yaw={et.get_yaw():.2f}, start_yaw={et.get_start_yaw():.2f}, error={et.get_yaw() - et.get_start_yaw():.2f}, left_speed={left_speed:.2f}, right_speed={right_speed:.2f}")
                et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
            elif prev_mode_test:
                target_yaw = None
                prev_mode_test = False
            elif mode == Mode.PAUSE:
                left_speed, right_speed = 0, 0
                et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
                # 急停止フラグが残っていればゼロ速度出力＆フラグクリア
                if turn_left_started == 'reverse_stop_left':
                    turn_left_started = False
                if turn_right_started == 'reverse_stop_right':
                    turn_right_started = False

            # --- ループ周期制限とdebug出力（最後） ---
            loop_end = time.time()
            handle_debug_output(loop_start, loop_end, debug_state)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        et.stop()
        # videoを使用している場合は解放
        video.release()

        # 録画スレッドの停止とクリーンアップ
        if save_camera_video and video_writer is not None:
            video_thread_running = False
            if video_thread is not None:
                video_thread.join(timeout=2)
            video_writer.release()
            if video_filename:
                print(f"Video saved to: {video_filename}")

        # Clean up sensor recorder if it was used
        if record_sensor_data and sensor_recorder is not None:
            sensor_recorder.stop_recording()
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")
        # Restore terminal settings on exit
        keyboard.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OpenCV-based line following robot with optional sensor recording and video saving")
    parser.add_argument("--record-sensor", action="store_true", help="Record sensor data to file")
    parser.add_argument("--save-video", action="store_true", help="Save camera video to file")
    parser.add_argument("--course", choices=["left", "right"], default="right", help="Initial course to follow: 'left' for left edge, 'right' for right edge (default: right)")
    parser.add_argument("--course-type", choices=["upper", "lower"], default="upper", help="Course type: 'upper' or 'lower' (default: upper)")
    parser.add_argument("--manual", action="store_true", help="Enable manual key input control mode")
    args = parser.parse_args()
    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_CNN}")
    print(f"Base speed: {BASE_SPEED}")
    print(f"course: {args.course}")
    print("Controls:")
    print("  'a' - Follow left edge")
    print("  'd' - Follow right edge")
    print("  'f' - Forward")
    print("  'b' - Backward")
    print("  'q' - Quit")

    main(
        record_sensor_data=args.record_sensor,
        save_camera_video=args.save_video,
        course=args.course,
        course_type=args.course_type,
        manual_mode=args.manual,
    )
