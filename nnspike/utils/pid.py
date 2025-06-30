import time
import csv
import os
from typing import Optional, Tuple


class PIDController:
    """
    PID（比例・積分・微分）コントローラは、産業用制御システムで広く使われる制御ループ機構です。
    このクラスは基本的なPIDコントローラを実装します。

    属性:
        Kp (float): 比例ゲイン。
        Ki (float): 積分ゲイン。
        Kd (float): 微分ゲイン。
        setpoint (float): システムが目指す目標値。
        output_limits (tuple[int, int]): 出力の最小値と最大値。
        _last_error (float): 前回の誤差。
        _integral (float): 誤差の積分値。
    """

    # 例：output/pid_logs/pid_debug.csv に保存する場合
    # pid = PIDController(
    #     Kp=1.0,
    #     Ki=0.0,
    #     Kd=0.1,
    #     setpoint=100,
    #     debug_log_path="output/pid_logs/pid_debug.csv"
    # )
    #
    # 絶対パス例：
    # pid = PIDController(
    #     Kp=1.0,
    #     Ki=0.0,
    #     Kd=0.1,
    #     setpoint=100,
    #     debug_log_path="C:/Users/YourName/Documents/pid_debug.csv"
    # )

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        setpoint: float,
        output_limits: Tuple[Optional[float], Optional[float]] = (None, None),
        debug_log_path: Optional[str] = "output/pid/pid_debug.csv",
    ):
        """
        指定したゲイン、目標値、出力制限でPIDControllerを初期化します。

        引数:
            Kp (float): 比例ゲイン。
            Ki (float): 積分ゲイン。
            Kd (float): 微分ゲイン。
            setpoint (float): システムが目指す目標値。
            output_limits (tuple[float, float], optional): 出力の最小値と最大値。デフォルトは (None, None)。
        """
        self._debug_log_path = "output/pid/pid_debug.csv"
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._last_time = None
        self._last_error = 0.0
        self._integral = 0.0
        self._debug_log_path = debug_log_path
        self._debug_log = []

    def set_output_limits(self, new_output_limits: Tuple[Optional[float], Optional[float]]):
        """
        出力制限を新しく設定します。
        """
        self.output_limits = new_output_limits

    def update(self, measured_value: float) -> float:
        """
        現在値から制御量を計算します。

        引数:
            measured_value (float): 現在のプロセス値。

        戻り値:
            float: 制御出力（例：ステアリング角など）。
        """
        current_time = time.time()
        error = self.setpoint - measured_value  # クロストラック誤差

        if self._last_time is None:
            self._last_time = current_time
            self._last_error = error
            return 0.0  # 初回は0を返す

        delta_time = current_time - self._last_time
        delta_error = error - self._last_error

        # 積分項（アンチワインドアップ対応）
        self._integral += error * delta_time
        integral_min, integral_max = -float('inf'), float('inf')
        if self.output_limits[0] is not None:
            integral_min = self.output_limits[0]
        if self.output_limits[1] is not None:
            integral_max = self.output_limits[1]
        self._integral = max(integral_min, min(self._integral, integral_max))

        derivative = delta_error / delta_time if delta_time > 0 else 0

        # PID出力の計算
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        # デバッグ用にデータを記録
        if self._debug_log_path is not None:
            # ディレクトリが存在しない場合は作成
            dir_path = os.path.dirname(os.path.abspath(self._debug_log_path))
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            self._debug_log.append([
                current_time,
                measured_value,
                self.setpoint,
                error,
                self._integral,
                derivative,
                output
            ])
            # 100サンプルごとにCSVへ書き出し
            if len(self._debug_log) >= 100:
                with open(self._debug_log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(self._debug_log)
                self._debug_log = []

        # 出力制限の適用
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        # 状態の更新
        self._last_time = current_time
        self._last_error = error

        return output

    def flush_debug_log(self):
        """
        残っているデバッグデータをCSVに書き出す
        """
        if self._debug_log_path is not None and self._debug_log:
            dir_path = os.path.dirname(os.path.abspath(self._debug_log_path))
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            with open(self._debug_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self._debug_log)
            self._debug_log = []
