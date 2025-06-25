import serial
import time
import serial.tools.list_ports

# 利用可能なCOMポート一覧を表示
print('--- 利用可能なCOMポート一覧 ---')
for port in serial.tools.list_ports.comports():
    print(f'  {port.device} : {port.description}')
print('-----------------------------')

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
