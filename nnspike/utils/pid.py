"""
PID制御モジュール

このモジュールは、汎用的なPID（比例・積分・微分）制御アルゴリズムを実装します。
主にロボットや各種制御系のフィードバック制御に利用されます。

クラス:
    PIDController:
        比例（P）、積分（I）、微分（D）制御を組み合わせたコントローラ。
        ゲイン・出力制限・積分制限・微分ローパスフィルタ・デバッグ出力などをサポート。
"""

import time
from typing import Optional, Tuple
import json

# ==== ユーザー調整用PIDパラメータ（ここだけ編集すればOK）====
PID_KP = 0.6    # さらに曲がりやすさUP（0.55→0.60）
PID_KI = 0.0    # 積分は微小に
PID_KD = 0.22   # 微分をさらに弱めてふり幅抑制（0.28→0.22）
PID_SETPOINT = 0
PID_OUTPUT_LIMITS = (-0.32, 0.32)  # 出力幅もさらに抑制（-0.35,0.35→-0.32,0.32）
PID_DERIVATIVE_LPF_ALPHA = 0.9  # 応答性重視で0.9
PID_INTEGRAL_LIMITS = (None, None)


class PIDController:
    """
    汎用PID制御器。
    """

    def __init__(
        self,
        Kp: float = PID_KP,
        Ki: float = PID_KI,
        Kd: float = PID_KD,
        setpoint: float = PID_SETPOINT,
        output_limits: Tuple[Optional[float], Optional[float]] = PID_OUTPUT_LIMITS,
        derivative_lpf_alpha: float = PID_DERIVATIVE_LPF_ALPHA,
        integral_limits: Tuple[Optional[float], Optional[float]] = PID_INTEGRAL_LIMITS,
        debug: bool = True,
    ):
        """
        PIDControllerの初期化。

        引数:
            Kp: float 比例ゲイン
            Ki: float 積分ゲイン
            Kd: float 微分ゲイン
            setpoint: float 目標値
            output_limits: tuple 出力の最小・最大値
            derivative_lpf_alpha: float 微分項ローパスフィルタ係数
            integral_limits: tuple 積分項の制限
            debug: bool デバッグ出力ON/OFF
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._last_time = None
        self._last_error = 0.0
        self._integral = 0.0
        self.derivative_lpf_alpha = derivative_lpf_alpha
        self._last_derivative = 0.0
        self.integral_limits = integral_limits
        self.debug = debug  # デバッグ出力ON/OFF

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
            if self.debug:
                print("[PID] time\tmeasured\tsetpoint\terror\tintegral\traw_deriv\tderiv_lpf\toutput")
                print(f"[PID] {current_time}\t{measured_value}\t{self.setpoint}\t{error}\t0\t0\t0\t0")
            return 0.0  # 初回は0を返す

        delta_time = current_time - self._last_time
        delta_error = error - self._last_error

        # --- 積分項（アンチワインドアップ: 独立したintegral_limitsを優先） ---
        self._integral += error * delta_time
        integral_min, integral_max = -float('inf'), float('inf')
        if self.integral_limits[0] is not None:
            integral_min = self.integral_limits[0]
        if self.integral_limits[1] is not None:
            integral_max = self.integral_limits[1]
        self._integral = max(integral_min, min(self._integral, integral_max))

        # --- 微分項（ローパスフィルタ適用, delta_time極小時は0） ---
        if delta_time > 1e-4:
            raw_derivative = delta_error / delta_time
        else:
            raw_derivative = 0.0
        alpha = self.derivative_lpf_alpha
        derivative = alpha * raw_derivative + (1 - alpha) * self._last_derivative
        self._last_derivative = derivative

        # PID出力の計算
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        # デバッグ用にデータをprint表示（1行JSON形式, Copilotフレンドリー）
        if self.debug:
            debug_data = {
                "time": round(current_time, 3),
                "measured": round(measured_value, 4),
                "setpoint": round(self.setpoint, 4),
                "error": round(error, 4),
                "integral": round(self._integral, 4),
                "raw_derivative": round(raw_derivative, 4),
                "derivative": round(derivative, 4),
                "output": round(output, 4)
            }
            print(f"[PID_DEBUG] {json.dumps(debug_data, ensure_ascii=False)}")

        # 出力制限の適用
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        # 状態の更新
        self._last_time = current_time
        self._last_error = error

        return output


class ControlCalculator:
    def __init__(self, image_width, debug=True):
        # ...既存のコード...
        self.debug = debug  # デバッグ出力ON/OFF
