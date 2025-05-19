import socket
import struct
import pickle
import threading
import math
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation

MCAST_GRP = "224.1.1.1"
MCAST_PORT = 5005

# Camera and screen settings
width = 1920
height = 1080
circle_center_x = width // 2
circle_center_y = height // 2
circle_radius = 60


all_x_points = []
all_y_points = []


avg_x_points = []
avg_y_points = []

def receive_raw_points():
    # Set up multicast socket for raw points
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", MCAST_PORT))
    mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    while True:
        data = sock.recv(10240)
        data = pickle.loads(data)  
        x_points, y_points = data  
        print(f"Raw X_points: {x_points}, Y_points: {y_points}")
        all_x_points.append(x_points)
        all_y_points.append(y_points)

def receive_avg_points():
    # Set up multicast socket for averaged points
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", MCAST_PORT))
    mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # Temporary lists to calculate moving average
    temp_x = deque(maxlen=1)
    temp_y = deque(maxlen=1)

    while True:
        data = sock.recv(10240)
        data = pickle.loads(data)  
        x_points, y_points = data  
        
        # Add to temporary lists
        temp_x.append(x_points)
        temp_y.append(y_points)

        avg_x = sum(temp_x) / len(temp_x)
        avg_y = sum(temp_y) / len(temp_y)
        avg_x_points.append(avg_x)
        avg_y_points.append(avg_y)
        print(f"Avg X: {avg_x:.2f}, Avg Y: {avg_y:.2f}")


raw_thread = threading.Thread(target=receive_raw_points)
raw_thread.daemon = True
raw_thread.start()

avg_thread = threading.Thread(target=receive_avg_points)
avg_thread.daemon = True
avg_thread.start()

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(0, width)
ax.set_ylim(0, height)
raw_scatter = ax.scatter([], [], c='blue', s=10, label='Raw Points')
avg_scatter = ax.scatter([], [], c='red', s=8, label='10-Point Average')

# Draw the circle
circle = plt.Circle((circle_center_x, circle_center_y), circle_radius, 
                   fill=False, color='red')
ax.add_artist(circle)

def update(frame):
    # Update raw points
    raw_scatter.set_offsets(list(zip(all_x_points, all_y_points)))
    # Update averaged points
    avg_scatter.set_offsets(list(zip(avg_x_points, avg_y_points)))
    return raw_scatter, avg_scatter

# Create animation
ani = FuncAnimation(fig, update, interval=100, blit=True)

# Show the plot
plt.title("Raw and Averaged Points Plot")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid(True)
plt.legend()
plt.show()



        



































# import socket
# import struct
# import pickle
# import threading
# import math
# from collections import deque


# MCAST_GRP = "224.1.1.1"
# MCAST_PORT = 5005

# # Camera and screen settings
# width = 1920
# height = 1080
# circle_center_x = width // 2
# circle_center_y = height // 2
# circle_radius = 60


# cx = deque(maxlen=1)
# cy = deque(maxlen=1)


# def  recieve_points():
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
#         print(f"X_points :{x_points} ,Y_points:{y_points}")
#         cx.append(x_points)
#         cy.append(y_points)

# recieve_thread = threading.Thread(target=recieve_points)
# recieve_thread.start()