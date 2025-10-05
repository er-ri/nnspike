import time


class PIDController:
    """
    A PID (Proportional-Integral-Derivative) controller is a control loop mechanism widely used in industrial control systems.
    This class implements a basic PID controller.

    Attributes:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        setpoint (float): Desired value that the system should achieve.
        output_limits (tuple[int, int]): Minimum and maximum limits for the output.
        _previous_error (float): Error from the previous update, used to calculate the derivative term.
        _integral (float): Accumulated integral of the error over time.
    """

    def __init__(
        self,
        Kp: float = 45,  # デフォルトはBASE_SPEED相当
        Ki: float = 0,
        Kd: float = 5,   # デフォルトはBASE_SPEED * 0.11相当
        setpoint: float = 0,
        output_limits: tuple[float | None, float | None] = (-45, 45),  # デフォルトはBASE_SPEED相当
    ):
        """
        Initializes the PIDController with the specified gains, setpoint, and output limits.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            setpoint (float): Desired value that the system should achieve.
            output_limits (tuple[float, float], optional): Minimum and maximum limits for the output. Defaults to (None, None).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._last_time = time.time()
        self._last_error = 0.0
        self._integral = 0.0

    def set_output_limits(self, new_output_limits: tuple[float, float]):
        self.output_limits = new_output_limits

    def update(self, measured_value: float, base_speed: float) -> float:
        """
        Calculate the control variable based on the measured value.

        Args:
            measured_value (float): The current value of the process variable.
            base_speed (float): Base speed value to dynamically adjust Kp, Kd, and output_limits.

        Returns:
            float: Control output, typically used for wheel steering or other control mechanisms.
        """
        current_time = time.time()
        error = self.setpoint - measured_value  # Cross Track Error

        delta_time = current_time - self._last_time if self._last_time is not None else 0
        delta_error = error - self._last_error

        self._integral += error * delta_time
        derivative = delta_error / delta_time if delta_time > 0 else 0

        # Calculate PID output using base_speed for dynamic parameters
        kp = base_speed  # Proportional gain based on base_speed
        kd = base_speed * 0.1  # Derivative gain: 元の比率 5/50 = 0.1
        
        output = kp * error + self.Ki * self._integral + kd * derivative

        # Apply output limits based on base_speed
        if output > base_speed:
            output = base_speed
        elif output < -base_speed:
            output = -base_speed

        # Update state
        self._last_time = current_time
        self._last_error = error

        return output
