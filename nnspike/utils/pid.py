import time
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

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        setpoint: float,
        output_limits: Tuple[Optional[float], Optional[float]] = (None, None),
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
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._last_time = None
        self._last_error = 0.0
        self._integral = 0.0

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
            print(f"{current_time}\t{measured_value}\t{self.setpoint}\t{error}\t0\t0\t0")
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

        # デバッグ用にデータをprint表示
        print(f"{current_time}\t{measured_value}\t{self.setpoint}\t{error}\t{self._integral}\t{derivative}\t{output}")

        # 出力制限の適用
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        # 状態の更新
        self._last_time = current_time
        self._last_error = error

        return output
