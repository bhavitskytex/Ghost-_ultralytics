import socket
import struct
import pickle
import math
from  collections import deque
from relative_commands import SiyiMount
import time


MCAST_GRP = "224.1.1.1"
MCAST_PORT = 5005

# Camera and screen settings
width = 1920
height = 1080
circle_center_x = width // 2
circle_center_y = height // 2
circle_radius = 60

smoothing_x = deque(maxlen=10)
smoothing_y = deque(maxlen=10)


cam = SiyiMount()
cam.move_relative(0.0, 0.0) 
print("Moved to initial  position")

def gimble_movement():
    previous_x = 0
    previous_y = 0
    # Set up multicast socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", MCAST_PORT))
    mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    while True:
        data = sock.recv(10240)
        data = pickle.loads(data)  
        x_points, y_points = data

        smoothing_x.append(x_points)
        smoothing_y.append(y_points)

        avg_x = int(sum(smoothing_x)/len(smoothing_x))
        avg_y = int(sum(smoothing_y)/len(smoothing_y))


        print("Average points : " ,avg_x ,avg_y)


        if avg_x != previous_x or avg_y != previous_y:
                print(f"Moving gimble")
                previous_x = avg_x
                previous_y = avg_y



        else:
            print("Else no need to move")
          





gimble_movement()
































# import socket
# import struct
# import pickle
# import math
# from relative_commands import SiyiMount

# MCAST_GRP = "224.1.1.1"
# MCAST_PORT = 5005

# # Camera and screen settings
# width = 1920
# height = 1080
# circle_center_x = width // 2
# circle_center_y = height // 2
# circle_radius = 60


# cam = SiyiMount()
# cam.move_relative(0.0, 0.0) 
# print("Moved to initial  position")

# def gimble_movement():
#     # Set up multicast socket
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
#     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     sock.bind(("", MCAST_PORT))
#     mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
#     sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

#     while True:
#         data = sock.recv(10240)
#         data = pickle.loads(data)  
#         x_points, y_points = data  
#         print(data)

        
#         distance = math.sqrt((x_points - circle_center_x) ** 2 + (y_points - circle_center_y) ** 2)

#         # Move the camera only if the point is outside the defined radius
#         if distance > circle_radius:
#             # pitch, yaw = cam.calculate_gimbal_angles(x_points, y_points, width, height)
#             # cam.move_relative(pitch, yaw)
#             # print(f"MOVING CAM to Pitch: {pitch}, Yaw: {yaw}")
#             print("Cam moving--------------")

# gimble_movement()