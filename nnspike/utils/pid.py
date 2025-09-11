"""PID (Proportional-Integral-Derivative) controller implementation for control systems.

This module provides a discrete-time PID controller class designed for real-time
control applications in robotics, automation, and other feedback control systems.
The implementation focuses on practical usage with features like output limiting,
time-aware calculations, and integral windup prevention.

**Methodology:**
The PID controller implements the standard control algorithm:
    output = Kp * error + Ki * ∫error*dt + Kd * d(error)/dt

Where:
- Kp (Proportional): Reacts to current error magnitude
- Ki (Integral): Eliminates steady-state error by accumulating past errors
- Kd (Derivative): Dampens oscillations by predicting future error trends

**Results:**
The controller produces smooth, stable control signals suitable for:
- Motor speed and position control
- Temperature regulation
- Line following in robotics
- Steering correction systems
- Process control applications

**Conclusion:**
This implementation provides a robust foundation for feedback control systems
with proper time handling and output constraints. Users should focus on proper
parameter tuning (Kp, Ki, Kd) for their specific application to achieve optimal
performance and system stability.

Example:
    Basic usage for a line-following robot:

    >>> from nnspike.utils.pid import PIDController
    >>>
    >>> # Create steering controller (setpoint = 0 means centered)
    >>> steering_pid = PIDController(kp=1.2, ki=0.05, kd=0.8, setpoint=0.0)
    >>> steering_pid.set_output_limits((-1.0, 1.0))  # Limit steering range
    >>>
    >>> # Control loop
    >>> while robot.is_running():
    ...     line_position = robot.get_line_position()  # -1 to 1 range
    ...     steering_correction = steering_pid.update(line_position)
    ...     robot.set_steering(steering_correction)
    ...     time.sleep(0.01)  # 100Hz control frequency

Classes:
    PIDController: Main PID controller implementation with output limiting.

Dependencies:
    time: For delta time calculations between updates.
"""

from __future__ import annotations

import time


class PIDController:
    """A PID (Proportional-Integral-Derivative) controller for closed-loop control systems.

    This class implements a discrete-time PID controller that continuously calculates
    an error value as the difference between a desired setpoint and a measured process
    variable. The controller applies proportional, integral, and derivative corrections
    to minimize this error over time, making it suitable for robotic control systems,
    motor speed regulation, and other feedback control applications.

    **Methodology:**
    The PID controller uses three distinct parameters:
    - Proportional (P): Provides output proportional to the current error
    - Integral (I): Accumulates past errors to eliminate steady-state error
    - Derivative (D): Predicts future error based on the rate of change

    The control output is calculated as:
    output = Kp * error + Ki * integral + Kd * derivative

    where error = setpoint - measured_value

    **Results:**
    The controller produces a continuous control signal that drives the system
    towards the desired setpoint. Output can be constrained within specified
    limits to prevent actuator saturation or system damage.

    **Conclusion:**
    This implementation provides time-aware PID control with integral windup
    prevention through output limiting. The controller maintains internal state
    between updates, making it suitable for real-time control applications.
    Note that proper tuning of Kp, Ki, and Kd parameters is crucial for optimal
    performance and system stability.

    Attributes:
        kp (float): Proportional gain coefficient.
        ki (float): Integral gain coefficient.
        kd (float): Derivative gain coefficient.
        setpoint (float): Target value that the system should achieve.
        output_limits (tuple[float | None, float | None]): Min/max output constraints.

    Example:
        >>> # Create PID controller for steering control
        >>> pid = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=0.0)
        >>> pid.set_output_limits((-1.0, 1.0))
        >>>
        >>> # Update control loop
        >>> measured_position = get_sensor_reading()
        >>> control_output = pid.update(measured_position)
        >>> apply_control_signal(control_output)
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float,
        output_limits: tuple[float | None, float | None] = (None, None),
    ):
        """Initializes the PIDController with specified parameters.

        Sets up the PID controller with the given gain values, target setpoint,
        and optional output constraints. Initializes internal state variables
        for time tracking and error accumulation.

        Args:
            kp (float): Proportional gain coefficient. Higher values increase
                responsiveness but may cause overshoot.
            ki (float): Integral gain coefficient. Eliminates steady-state error
                but may cause instability if too high.
            kd (float): Derivative gain coefficient. Reduces overshoot and
                improves stability but sensitive to noise.
            setpoint (float): Target value that the controller should achieve.
            output_limits (tuple[float | None, float | None], optional):
                Minimum and maximum output bounds. Use None for unbounded limits.
                Defaults to (None, None).

        Example:
            >>> # Create PID for temperature control
            >>> pid = PIDController(
            ...     kp=2.0, ki=0.5, kd=0.1, setpoint=25.0, output_limits=(-100, 100)
            ... )
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
        """Updates the output limits for the PID controller.

        Modifies the minimum and maximum bounds for the controller output.
        This helps prevent actuator saturation and integral windup.

        Args:
            new_output_limits (tuple[float, float]): New minimum and maximum
                output limits as (min_limit, max_limit).

        Example:
            >>> pid.set_output_limits((-50.0, 50.0))  # Limit output to ±50
        """
        self.output_limits = new_output_limits

    def update(self, measured_value: float) -> float:
        """Computes the PID control output based on current measurement.

        Calculates the control signal by combining proportional, integral, and
        derivative terms. Updates internal state for the next iteration and
        applies output limits if specified.

        Args:
            measured_value (float): Current value of the process variable being
                controlled (e.g., current position, temperature, speed).

        Returns:
            float: Control output signal, typically used for actuator control
            (e.g., motor commands, valve positions). Value is constrained
            within the specified output limits.

        Note:
            The first call may have reduced accuracy for the derivative term
            due to lack of previous time reference.

        Example:
            >>> current_temp = thermometer.read()
            >>> heater_power = pid.update(current_temp)
            >>> heater.set_power(heater_power)
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
