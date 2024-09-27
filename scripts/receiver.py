import cv2
import socket
import pickle
import struct
import time
import numpy as np

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 8485))
server_socket.listen(0)

conn, addr = server_socket.accept()

data = b""
payload_size = struct.calcsize("Q")  # Use "Q" for unsigned long long (8 bytes)

FILE_LABEL = time.strftime("%Y%m%d%H%M%S", time.localtime())

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(f"storage/record/{FILE_LABEL}.avi", fourcc, 20.0, (640, 480))

while True:
    while len(data) < payload_size:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

    if len(data) < payload_size:
        break

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

    if len(data) < msg_size:
        break

    frame_data = data[:msg_size]
    data = data[msg_size:]

    try:
        img_encoded = pickle.loads(frame_data)
    except pickle.UnpicklingError as e:
        print(f"UnpicklingError: {e}")
        continue

    nparr = np.frombuffer(img_encoded, np.uint8)

    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Write the frame to the video file
    out.write(frame)

    cv2.imshow("Robot View", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything when job is finished
conn.close()
out.release()
cv2.destroyAllWindows()
