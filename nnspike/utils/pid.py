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
        kp: float,
        ki: float,
        kd: float,
        setpoint: float,
        output_limits: tuple[float | None, float | None] = (None, None),
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
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._last_time = time.time()
        self._last_error = 0.0
        self._integral = 0.0

    def set_output_limits(self, new_output_limits: tuple[float, float]) -> None:
        """
        Set new output limits for the PID controller.

        Args:
            new_output_limits (tuple[float, float]): The new minimum and maximum output limits.
        """
        self.output_limits = new_output_limits

    def update(self, measured_value: float) -> float:
        """
        Calculate the control variable based on the measured value.

        Args:
            measured_value (float): The current value of the process variable.

        Returns:
            float: Control output, typically used for wheel steering or other control mechanisms.
        """
        current_time = time.time()
        error = self.setpoint - measured_value  # Cross Track Error

        delta_time = (
            current_time - self._last_time if self._last_time is not None else 0
        )
        delta_error = error - self._last_error

        self._integral += error * delta_time
        derivative = delta_error / delta_time if delta_time > 0 else 0

        # Calculate PID output
        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        # Update state
        self._last_time = current_time
        self._last_error = error

        return output
