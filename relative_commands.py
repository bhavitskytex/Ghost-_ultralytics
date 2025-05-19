import socket
import struct
import time

 
class SiyiMount:
    # Command constants
    GET_ATTITUDE = bytes.fromhex("55 66 01 00 00 00 00 0d e8 05")
    CENTER = bytes.fromhex("55 66 01 01 00 00 00 08 01 d1 12")
 
    def __init__(self, ip="192.168.144.25", port=37260):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(5)
        self.current_yaw = 0
        self.current_pitch = 0
 
    def __del__(self):
        if self.sock:
            self.sock.close()
 
    def send_command(self, command):
        try:
            self.sock.sendto(command, (self.ip, self.port))
            return self.sock.recvfrom(1024)[0]
        except socket.timeout:
            return None
 
    def get_attitude(self):
        response = self.send_command(self.GET_ATTITUDE)
        if response and len(response) >= 14:
            print("INSIDE RESPONSE")
            yaw, pitch, roll = struct.unpack('<hhh', response[8:14])
            self.current_yaw = -yaw * 0.1
            self.current_pitch = pitch * 0.1
            return self.current_yaw, self.current_pitch, roll * 0.1
        print("NO RESPONSE")
        return None
 
    def center(self):
        response = self.send_command(self.CENTER)
        if response:
            self.current_yaw = 0
            self.current_pitch = 0
            return True
        return False
 
    def move_relative(self, rel_pitch_deg, rel_yaw_deg):
        new_yaw = self.current_yaw + rel_yaw_deg
        new_pitch = self.current_pitch + rel_pitch_deg
        
        new_yaw = max(min(new_yaw, 180), -180)
        new_pitch = max(min(new_pitch, 90), -90)
        
        yaw_cdeg = int(-new_yaw * 10)
        pitch_cdeg = int(new_pitch * 10)
        
        command = struct.pack('<BBBHHBHH', 0x55, 0x66, 0x01, 0x0004, 0x0000, 0x0E,
                              yaw_cdeg & 0xFFFF, pitch_cdeg & 0xFFFF)
        command += struct.pack('<H', self.crc16_ccitt(command))
        
        response = self.send_command(command)
        print("sending command to change position",response)
        if response:
            self.current_yaw = new_yaw
            self.current_pitch = new_pitch
            return True
        return False
    
    def calculate_gimbal_angles(self, x, y, frame_width, frame_height):
        # Unpack values
        frame_x = frame_width / 2
        frame_y = frame_height / 2
        obj_x, obj_y = x, y
        width, height = (frame_width, frame_height)
        hfov, vfov = (81, 65.60)  # Check the camera specs
 
        # Calculate offsets
        offset_x = obj_x - frame_x
        offset_y =(frame_y - obj_y ) # Invert y-axis
 
        # Calculate yaw and pitch angles
        yaw = (offset_x / (width / 2)) * (hfov / 2)
        pitch = (offset_y / (height / 2)) * (vfov / 2)
 
        print(yaw, pitch)
        return pitch, yaw
 
    @staticmethod
    def crc16_ccitt(data):
        crc = 0
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                crc = (crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1
        return crc & 0xFFFF