# LEGO type:standard slot:2 autostart
"""Main controlling program for LEGO Spike Prime Hub 

The program was running on a LEGO Spike Prime Hub that controlled the 
ETRobot's behavior by receiving commands that were sent by the Raspberry Pi 
through GPIO. The asynchronous communication with Raspberry Pi was 
implemented by the micropython package `uasyncio`.  

Appendix:

**Color Sensor**

The Color Sensor can give the following data:

| Mode | Output | Description |
| --- | --- | --- |
| Color(2) | -1 ~ 10 | Each number represents a detected color |
| Reflected(3) | 0 ~ 100 | 0% = no reflection, 100% = very reflective |
| Ambient(2) | 0 ~ 100 | 0% = dark, 100% = bright |

Set the mode and retrieve the sensor data in the following way

```python
# Set mode
color_sensor.mode(0)
# Retrieve its detected value
ret = color_sensor.get()
```

Ref: https://ev3-help-online.api.education.lego.com/Retail/en-us/page.html?Path=editor%2FUsingSensors_Color.html

"""
import gc
import hub
import time
import uasyncio

MAX_IDEL_TIME = 120000  # Maximum idel time, unit: millisecond
MAX_RUN_TIME = 600  # Maximum running time, unit: second

# Command ID list
COMMAND_MOTOR_ID = 201
COMMAND_ARM_ID = 202
COMMAND_RESET_MOTOR_ID = 203

PORT_MAP = {
    "motor_arm": "F",
    "motor_right": "C",
    "motor_left": "B",
    "color_sensor": "E",
    "ultrasonic_sensor": "A",
    "serial_port": "D",
}


class LegoSpike(object):
    """LEGO Spike Prime Hub

    Class for controlling all the devices in the Spike car by receiving
    commands from Raspberry Pi. Every command is made up of 2 bytes, the
    first byte indicates the command id while the second byte represents
    the corresponding parameters as shown below.

    | Device | Command Id | Parameter1 | Parameter2 |
    | Motor | 0 | Power: 0~180 | Steering: -90 ~ 90  |

    """

    def __init__(self) -> None:
        # Initialization
        hub.display.show(
            hub.Image.ALL_CLOCKS, delay=400, clear=True, wait=False, loop=True, fade=0
        )
        hub.motion.align_to_model(hub.TOP, hub.FRONT)  # GYRO, orientation
        hub.motion.yaw_pitch_roll(0)  # yaw, pitch and roll

        # Set ports
        self.motor_arm = getattr(hub.port, PORT_MAP["motor_arm"]).motor
        self.motor_right = getattr(hub.port, PORT_MAP["motor_right"]).motor
        self.motor_left = getattr(hub.port, PORT_MAP["motor_left"]).motor
        self.color_sensor = getattr(hub.port, PORT_MAP["color_sensor"]).device
        self.ultrasonic_sensor = getattr(hub.port, PORT_MAP["ultrasonic_sensor"]).device
        self.serial_port = getattr(hub.port, PORT_MAP["serial_port"])

        # The serial port will operates as a raw full duplex logic level(data can be transmitted in both directions send/receive)
        self.serial_port.mode(hub.port.MODE_FULL_DUPLEX)
        # Setup the serial port(take 1 second), and transferring a maximum of 115200 bits per second.
        time.sleep(1)
        self.serial_port.baud(115200)

        # Initialization & Set motors mode to measure its relative position on boot
        self.motor_left.mode([(2, 0)])
        self.motor_right.mode([(2, 0)])
        self.motor_left.preset(0)
        self.motor_right.preset(0)

        # Set color sensor to reflection mode
        self.color_sensor.mode(1)

        # Set ultrasonic sensor mode to return "Distance Centimeters"
        self.ultrasonic_sensor.mode(0)

        # Clear serial port buffer
        while self.serial_port.read(100) != b"":
            continue

        # Millisecond counter for record the latest command executed time, maximum idle time
        self.command_counter = time.ticks_ms()

        hub.display.show(hub.Image.HEART)

    def read_command(self):
        raw_bytes = self.serial_port.read(3)
        command_id = None
        command_parameter1 = None
        command_parameter2 = None

        if raw_bytes != b"":
            command_id = int.from_bytes(raw_bytes[0:1], "big")
            command_parameter1 = int.from_bytes(raw_bytes[1:2], "big")
            command_parameter2 = int.from_bytes(raw_bytes[2:3], "big")
            self.command_counter = time.ticks_ms()

        return command_id, command_parameter1, command_parameter2

    def send_status(self):
        """
        Sends the current status of the robot to a connected serial port.

        This method retrieves the reflection value from the color sensor and the
        motor counts from the left and right motors. It then processes these values
        to create a status message that is sent via the serial port.

        The reflection value is converted to a single byte, and the motor count is
        normalized to semicircular units (by dividing by 180) and converted to two bytes.
        These bytes are concatenated and written to the serial port.

        Note:
            Sent data are following the below format

            12233

            Where '1' is the reflection(color sensor), '2' is the motor count
        """
        # Color sensor
        reflection = self.color_sensor.get()[0]

        # Motor count
        motor_left_count = self.motor_left.get()[0] * -1
        motor_right_count = self.motor_right.get()[0]
        motor_count = motor_left_count + motor_right_count
        motor_count = motor_count // 360

        # Ultrasonic sensor
        distance = self.ultrasonic_sensor.get()[0]
        distance = distance if distance is not None else 200

        # Type cast to bytes
        reflection_byte = reflection.to_bytes(1, "big")
        motor_count_byte = motor_count.to_bytes(2, "big")
        distance_byte = distance.to_bytes(1, "big")

        status_data = reflection_byte + motor_count_byte + distance_byte
        self.serial_port.write(status_data)

    def execute_command(self, command_id, command_parameter1, command_parameter2):

        print(
            "Command received, id={}; parameter1={}; parameter2={}".format(
                command_id, command_parameter1, command_parameter2
            )
        )

        if command_id == COMMAND_MOTOR_ID:
            self._set_motor_power(command_parameter1, command_parameter2)
        elif command_id == COMMAND_RESET_MOTOR_ID:
            self._reset_motor()
        elif command_id == COMMAND_ARM_ID:
            pass

    def _set_motor_power(self, left_power: int, right_power: int) -> None:
        """Method to control the steering wheel angle.

        Args:
            left_power: Left wheel power(0~100)
            right_power: Right wheel power(0~100)
        """
        self.command_counter = time.ticks_ms()

        self.motor_left.pwm(-int(left_power))
        self.motor_right.pwm(int(right_power))

    def _reset_motor(self) -> None:
        self.command_counter = time.ticks_ms()

        self.motor_left.preset(0)
        self.motor_right.preset(0)


async def receiver():
    while True:
        command_id, command_parameter1, command_parameter2 = lego_spike.read_command()
        if command_id != None:
            lego_spike.execute_command(
                command_id, command_parameter1, command_parameter2
            )

        if time.ticks_ms() - lego_spike.command_counter > MAX_IDEL_TIME:
            raise SystemExit("Maximum idle time reached, terminate lego spike.")

        await uasyncio.sleep(0)


async def sender():
    while True:
        lego_spike.send_status()
        await uasyncio.sleep(0.005)


async def main_task():
    tasks = list()

    receiver_task = uasyncio.create_task(receiver())
    tasks.append(receiver_task)
    sender_task = uasyncio.create_task(sender())
    tasks.append(sender_task)

    await uasyncio.sleep(MAX_RUN_TIME)

    # Cancel all tasks.
    for task in tasks:
        task.cancel()


# Trigger a garbage collection cycle
gc.collect()

print("Starting LEGO Prime Hub..")

try:
    lego_spike = LegoSpike()
    uasyncio.run(main_task())
except SystemExit as e:
    print(e)

lego_spike.motor_left.brake()
lego_spike.motor_right.brake()

hub.display.show(hub.Image.ASLEEP)

print("Ended")
