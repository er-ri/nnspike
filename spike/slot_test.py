import gc
import time

import hub  # type: ignore
import uasyncio  # type: ignore

CMD_FLAG = b"CF:"

usb = hub.USB_VCP()
usb.setinterrupt(-1)

# if usb.isconnected():
#     hub.display.show(hub.Image.HAPPY)

motor_right = hub.port.A.motor
motor_left = hub.port.B.motor


async def receiver() -> None:
    """Receiver task to handle incoming USB commands."""
    while True:
        if usb.any():
            data = usb.read(6)  # Increased to read "ET:" + 3 bytes command data

            et_pos = data.find(CMD_FLAG)
            if et_pos >= 0 and len(data) >= et_pos + 6:  # Ensure we have enough bytes
                raw_bytes = data[
                    et_pos + 3 : et_pos + 6
                ]  # Extract the 3 bytes after "CF:"
                command_id = int.from_bytes(raw_bytes[0:1], "big")
                command_parameter1 = int.from_bytes(raw_bytes[1:2], "big")
                command_parameter2 = int.from_bytes(raw_bytes[2:3], "big")

                # Handle different commands based on command_id
                if command_id == 201:  # Forward
                    hub.display.show(hub.Image.HAPPY)
                    motor_left.run_at_speed(-command_parameter1)
                    motor_right.run_at_speed(command_parameter2)
                elif command_id == 202:  # Backward
                    hub.display.show(hub.Image.SAD)
                    motor_left.run_at_speed(command_parameter1)
                    motor_right.run_at_speed(-command_parameter2)
                elif command_id == 203:  # Stop
                    hub.display.show(hub.Image.ASLEEP)
                    motor_left.pwm(0)
                    motor_right.pwm(0)

        else:
            hub.display.show(hub.Image.SILLY)

        await uasyncio.sleep(0)


async def main_task() -> None:
    tasks = []

    receiver_task = uasyncio.create_task(receiver())
    tasks.append(receiver_task)

    await uasyncio.sleep(10)

    motor_left.pwm(0)
    motor_right.pwm(0)

    # Cancel all tasks.
    for task in tasks:
        task.cancel()


# Trigger a garbage collection cycle
gc.collect()

print("Starting LEGO Prime Hub..")
command_counter = time.ticks_ms()  # type: ignore

try:
    uasyncio.run(main_task())
except SystemExit as e:
    print(e)


hub.display.show(hub.Image.ASLEEP)

print("Ended")
