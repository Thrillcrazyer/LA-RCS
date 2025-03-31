import socket
import numpy as np
import cv2
import threading

def call_image(client_socket):
    client_socket.send(b'REQUEST_IMAGE')

def call_distance(client_socket):
    client_socket.send(b'REQUEST_DISTANCE')

def receive_distance(connection):
    distance_info = connection.recv(16)
    distance = float(distance_info.decode().strip())
    return distance

def receive_frame(connection):
    size_info = connection.recv(16)
    size = int(size_info.decode().strip())
    
    data = b''
    while len(data) < size:
        packet = connection.recv(4096)
        if not packet:
            break
        data += packet

    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

def get_data(conn, command):
    if command == "IMAGE":
        call_image(conn)
        frame = receive_frame(conn)
        if frame is not None:
            # Optionally resize the frame here if needed
            frame_resized = cv2.resize(frame, (800, int(frame.shape[0] * (800 / frame.shape[1]))))
            return frame_resized
        return None
    elif command == "DISTANCE":
        call_distance(conn)
        return receive_distance(conn)
    else:
        conn.sendall(command.encode('utf-8'))
    return None

def init_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 5000))
    server_socket.listen(1)
    print("서버 대기 중...")
    conn, addr = server_socket.accept()
    print(f"연결됨: {addr}")
    return conn

def main():
    conn = init_server()
    while True:
        
        command = input("Enter command: ").strip().upper()
        data=get_data(conn,command)
        
        if command in ["IMAGE", "DISTANCE"]:
            if command == "IMAGE" and data is not None:
                cv2.imshow('Received Frame', data)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            elif command == "DISTANCE":
                print(f"Received distance: {data} meters")

if __name__ == "__main__":
    main()