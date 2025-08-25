import select
import sys
import termios
import tty
from nnspike.constants import Mode

class KeyboardController:
    def __init__(self):
        self.running = True
        self.current_key = None
        self.old_settings = termios.tcgetattr(sys.stdin)  # type: ignore
        tty.setraw(sys.stdin.fileno())  # type: ignore

    def get_key(self):
        """Get a single keypress"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            key = sys.stdin.read(1).lower()
            return key
        return None

    def cleanup(self):
        """Restore terminal settings"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)  # type: ignore

    def get_mode_from_key(self, key, current_mode):
        """キー入力に応じてモードを返す。変更がなければcurrent_modeを返す"""
        # 以前のロジック：全モードでキーによるモード変更が可能
        if key is None:
            return current_mode
        if key == "q":
            self.running = False
            print("Quitting...")
            return Mode.PAUSE
        elif key == "a":
            print("Switched to following: left edge")
            return Mode.FOLLOW_LEFT_EDGE
        elif key == "d":
            print("Switched to following: right edge")
            return Mode.FOLLOW_RIGHT_EDGE
        elif key == "h":
            print("Switched to HIGH_SPEED mode")
            return Mode.HIGH_SPEED
        elif key == "l":
            print("Switched to turn left mode")
            return Mode.TURN_LEFT
        elif key == "f":
            print("Switched to forward mode")
            return Mode.FORWARD
        elif key == "j":
            print("Switched to small turn left mode")
            return Mode.SMALL_TURN_LEFT
        elif key == "k":
            print("Switched to small turn right mode")
            return Mode.SMALL_TURN_RIGHT
        elif key == "i":
            print("Switched to turn left (relative) mode")
            return Mode.TURN_LEFT_RELATIVE
        elif key == "o":
            print("Switched to turn right (relative) mode")
            return Mode.TURN_RIGHT_RELATIVE
        elif key == "b":
            print("Switched to backward mode")
            return Mode.BACKWARD
        elif key == "g":
            print("Switched to gate pass mode")
            return Mode.GATE_PASS
        elif key == "e":
            print("Switched to blue eyes mode")
            return Mode.EYE_BLUE
        elif key == "u":
            print("Switched to blue bottle catch mode")
            return Mode.BLUE_BOTTLE_CATCH
        elif key == "1":
            print("Switched to double loop mode")
            return Mode.DOUBLE_LOOP
        elif key == "2":
            print("Switched to obstacle avoidance mode")
            return Mode.AVOID_OBSTACLE
        elif key == "3":
            print("Switched to bottle carrying 1 mode")
            return Mode.CARRY_BOTTLE1
        elif key == "4":
            print("Switched to back and turn 1 mode")
            return Mode.BACK_AND_TURN1
        elif key == "5":
            print("Switched to bottle carrying 2 mode")
            return Mode.CARRY_BOTTLE2
        elif key == "6":
            print("Switched to back and turn 2 mode")
            return Mode.BACK_AND_TURN2
        elif key == "7":
            print("Switched to heading goal mode")
            return Mode.HEAD_GOAL
        elif key == "8" or key == "p":
            print("Pausing robot")
            return Mode.PAUSE
        return current_mode
