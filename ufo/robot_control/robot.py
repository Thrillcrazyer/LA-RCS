import socket
import numpy as np
import cv2

class robot_api(object):
    def __init__(self,port=5000):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(1)
        print("서버 대기 중...")
        conn, addr = server_socket.accept()
        print(f"연결됨: {addr}")
        self.conn = conn
        
    def _call_image(self):
        self.conn.send(b'REQUEST_IMAGE')
        
    def _call_distance(self):
        self.conn.send(b'REQUEST_DISTANCE')
    
    def _receive_distance(self):
        distance_info = self.conn.recv(16)
        distance = float(distance_info.decode().strip())
        return distance
    
    def _receive_frame(self):
        size_info = self.conn.recv(16)
        size = int(size_info.decode().strip())
        data = b''
        while len(data) < size:
            packet = self.conn.recv(4096)
            if not packet:
                break
            data += packet
             
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    def get_data(self,command):
        if command == "IMAGE":
            self._call_image()
            frame = self._receive_frame()
            if frame is not None:
                # Optionally resize the frame here if needed
                frame_resized = cv2.resize(frame, (800, int(frame.shape[0] * (800 / frame.shape[1]))))
                return frame_resized
            return None
        elif command == "DISTANCE":
            self._call_distance()
            return self._receive_distance()
        elif "SPEAKTEXT" in command:
            text=command.split("/")[1]
            print("말하는중 !!!!!!!!!!!!!!!-----------------------------------")
            print(text)
            
        else:
            self.conn.sendall(command.encode('utf-8'))
        return None
    
    def capture_frame(self, robot_save_path):
        """_summary_
        capture_screenshot_multiscreen 배껴놓은 코드
        호출하면 로봇한테서 Frame을 호출하고 save_path에 저장한다.
        """
        try:
            # 프레임을 캡처합니다
            frame = self.get_data("IMAGE")
            # 이미지를 저장합니다
            success = cv2.imwrite(robot_save_path, frame)
            # 저장 성공 여부에 따라 True 또는 False를 반환합니다
            return success
        except Exception as e:
            # 예외 발생 시 False를 반환합니다
            print(f"Error: {e}")
            return False
    
    def get_distance(self):
        try:
            # 프레임을 캡처합니다
            distance = self.get_data("DISTANCE")
            return distance
        except Exception as e:
            # 예외 발생 시 False를 반환합니다
            print(f"Error: {e}")
            return False

