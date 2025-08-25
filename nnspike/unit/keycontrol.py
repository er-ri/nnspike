import sys
import termios
import tty
import select
from nnspike.constants import Mode

class KeyboardController:
    def __init__(self):
        self.running = True
        self.current_key = None
        self.old_settings = termios.tcgetattr(sys.stdin)  # type: ignore
        tty.setraw(sys.stdin.fileno())  # type: ignore
        # 有効なモードキーリスト（run_manual.pyから移動）
        self._mode_keys = set([
            "a", "d", "h", "l", "f", "j", "k", "i", "o", "b", "g", "e", "u",
            "1", "2", "3", "4", "5", "6", "7", "8", "p", "n", "q"
        ])

    def get_key(self):
        """Get a single keypress"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            key = sys.stdin.read(1).lower()
            return key
        return None

    def is_mode_key(self, key):
        """
        有効なモードキーかどうか判定
        """
        return key in self._mode_keys

    def cleanup(self):
        """Restore terminal settings"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)  # type: ignore

    def get_mode_from_key(self, key):
        """キー入力からモードとメッセージを返す。quit判定も含む"""
        if key == "q":
            return "quit", "Quitting..."
        keymap = {
            "a": (Mode.FOLLOW_LEFT_EDGE, "Switched to following: left edge"),
            "d": (Mode.FOLLOW_RIGHT_EDGE, "Switched to following: right edge"),
            "h": (Mode.HIGH_SPEED, "Switched to HIGH_SPEED mode"),
            "l": (Mode.TURN_LEFT, "Switched to turn left mode"),
            "f": (Mode.FORWARD, "Switched to forward mode"),
            "j": (Mode.SMALL_TURN_LEFT, "Switched to small turn left mode"),
            "k": (Mode.SMALL_TURN_RIGHT, "Switched to small turn right mode"),
            "i": (Mode.TURN_LEFT_RELATIVE, "Switched to turn left (relative) mode"),
            "o": (Mode.TURN_RIGHT_RELATIVE, "Switched to turn right (relative) mode"),
            "b": (Mode.BACKWARD, "Switched to backward mode"),
            "g": (Mode.GATE_PASS, "Switched to gate pass mode"),
            "e": (Mode.EYE_BLUE, "Switched to blue eyes mode"),
            "u": (Mode.BLUE_BOTTLE_CATCH, "Switched to blue bottle catch mode"),
            "1": (Mode.DOUBLE_LOOP, "Switched to double loop mode"),
            "2": (Mode.AVOID_OBSTACLE, "Switched to obstacle avoidance mode"),
            "3": (Mode.CARRY_BOTTLE1, "Switched to bottle carrying 1 mode"),
            "4": (Mode.BACK_AND_TURN1, "Switched to back and turn 1 mode"),
            "5": (Mode.CARRY_BOTTLE2, "Switched to bottle carrying 2 mode"),
            "6": (Mode.BACK_AND_TURN2, "Switched to back and turn 2 mode"),
            "7": (Mode.HEAD_GOAL, "Switched to heading goal mode"),
            "8": (Mode.PAUSE, "Pausing robot"),
            "p": (Mode.PAUSE, "Pausing robot"),
        }
        return keymap.get(key, (None, None))
