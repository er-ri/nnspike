import serial
import time

# Windowsの場合はCOMポート名に変更（例: 'COM3'）
PORT = 'COM5'  # 適宜変更してください
BAUDRATE = 115200
TIMEOUT = 2

if __name__ == '__main__':
    try:
        with serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=TIMEOUT) as ser:
            print(f'Listening on {PORT} at {BAUDRATE}bps...')
            while True:
                data = ser.read_until(expected=b"\r") 
                if data:
                    print(data.decode(errors='ignore').strip())
                time.sleep(0.01)
    except serial.SerialException as e:
        print(f'Error: {e}')
    except KeyboardInterrupt:
        print('Exiting...')
