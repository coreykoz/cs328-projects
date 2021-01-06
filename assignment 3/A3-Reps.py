#
import math
import socket
import sys
import json
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
#import Queue
import numpy as np
import scipy as sp
from scipy.ndimage.interpolation import shift
from scipy.signal import find_peaks

# Replace the string with your user ID
user_id = "grp404"

"""
Define all variables used for the exercise detection algorithm.
You can modify these based on your algorithm.
"""
previousMagnitude = -1000000
previousPositive = False
previousRepTimestamp = 0

def detectReps(time,x_in,y_in,z_in):
    """
    Gyroscope-based Exercise Repetion Algorithm. In this function, use the dominant axis of motion to detect repeated motions of an arm
    performing a bicep curl. You could think of using either frequency domain analysis -- at what rate is the arm moving? Zero crossings --
    Can I detect when the arm transitions between flexion and extension motions or by integrating the sensor data -- how far has the arm
    moved and when should I decide that it is a flexion or an extension? You can also think of using time domain constraints to detect
    a sequence of alternating flexion and extension motions to help filter out other noise.
    """
    
    # TODO: Rep detection algorithm
    # example global variables - can be modified
    global previousMagnitude
    global previousPositive
    global previousRepTimestamp
    global smoothedvals
    global repindices

    global last_val, last_ext, last_flex
    global steps_indicies, ext_indicies, flex_indicies, ___
    global has_ext, has_flex
    global rep_count

    # Pseudo-code provided here:    
    #   if(rep_detected):
    #       print("REP_DETECTED at timestamp: " + str(timestamp))
    #   else:
    #       keep processing

    #plot each flexion and extension and print
    height_filter = 3
    steps_indicies, ___ = find_peaks(magvals, height=height_filter, prominence=.5)


    # counting reps
    # finding each extension
    y_inverse = yvals * (-1)
    ext_indicies, __ = find_peaks(y_inverse, height=height_filter, prominence = .5)
    if (len(ext_indicies) > last_ext):
        has_ext = True;
    last_ext = len(ext_indicies)

    # find each flexion
    flex_indicies, __ = find_peaks(yvals, height= height_filter, prominence = .5)
    if (len(flex_indicies) > last_flex):
        has_flex = True;
    last_flex = len(flex_indicies)

    # print each rep
    if (has_ext and has_flex):
        rep_count += 1
        has_ext = False
        has_flex = False
        print("REP_DETECTED at timestamp: " + str(time) + "REP COUNT: " + str(rep_count))

    return


#################   Server Connection Code  ####################

def authenticate(sock):
    """
    Authenticates the user by performing a handshake with the data collection server.
    
    If it fails, it will raise an appropriate exception.
    """
    msg_request_id = "ID"
    msg_authenticate = "ID,{}\n"
    msg_acknowledge_id = "ACK"
    
    message = sock.recv(256).strip().decode('ascii')
    if (message == msg_request_id):
        print("Received authentication request from the server. Sending authentication credentials...")
    else:
        print(type(message))
        print("Authentication failed!")
        raise Exception("Expected message {} from server, received {}".format(msg_request_id, message))

    sock.send(msg_authenticate.format(user_id).encode('utf-8'))

    try:
        message = sock.recv(256).strip().decode('ascii')
    except:
        print("Authentication failed!")
        raise Exception("Wait timed out. Failed to receive authentication response from server.")
        
    if (message.startswith(msg_acknowledge_id)):
        ack_id = message.split(",")[1]
    else:
        print("Authentication failed!")
        raise Exception("Expected message with prefix '{}' from server, received {}".format(msg_acknowledge_id, message))
    
    if (ack_id == user_id):
        print("Authentication successful.")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Authentication failed : Expected user ID '{}' from server, received '{}'".format(user_id, ack_id))
        
def recv_data():
    global receive_socket
    global t, x, y, z
    global tvals, xvals, yvals, zvals, smoothedvals, magvals, last_val, last_flex, last_ext, rep_count
    global has_ext, has_flex

    last_val = 0
    last_flex = 0
    last_ext = 0
    rep_count = 0
    has_ext = False
    has_flex = False
    
    previous_json = ''
    while True:
        try:
            message = receive_socket.recv(1024).strip().decode('ascii')
            json_strings = message.split("\n")
            json_strings[0] = previous_json + json_strings[0]
            for json_string in json_strings:
                try:
                    data = json.loads(json_string)
                except:
                    previous_json = json_string
                    continue
                previous_json = '' # reset if all were successful
                sensor_type = data['sensor_type']
                if (sensor_type == "SENSOR_GYRO"):
                    t=data['data']['t']
                    x=data['data']['x']
                    y=data['data']['y']
                    z=data['data']['z']

                    mag = math.sqrt(x**2 + y ** 2 + z**2)
                    
                    #Shift new data into the numpy plot buffers 
                    xvals = shift(xvals, 1, cval=0)
                    xvals[0] = x
                    
                    yvals = shift(yvals, 1, cval=0)
                    yvals[0] = y
                    
                    zvals = shift(zvals, 1, cval=0)
                    zvals[0] = z

                    #Calculated "magnitude"
                    magvals = shift(magvals, 1, cval=0)
                    magvals[0] = mag
                else:
                    print(data)
                
            sys.stdout.flush()
            detectReps(t,x,y,z)
        except KeyboardInterrupt: 
            # occurs when the user presses Ctrl-C
            print("User Interrupt. Quitting...")
            break
        
        except Exception as e:
            # ignore exceptions, such as parsing the json
            # if a connection timeout occurs, also ignore and try again. Use Ctrl-C to stop
            # but make sure the error is displayed so we know what's going on
            if (str(e) != "timed out"):  # ignore timeout exceptions completely       
                print(e)
            pass

# Helper function that animates the canvas
def animate(i):
    global tvals, xvals, yvals, zvals, magvals, repindices
    
    rep_marker_locs = np.nonzero(repindices)
    rep_marker_locs = list(rep_marker_locs[0])
    
    try:
        ax1.clear()
        ax2.clear()
        ax1.plot(tvals, xvals, label="X")
        ax1.plot(tvals, yvals, label="Y")
        ax1.plot(tvals, zvals, label="Z")
        ax1.set_title('Angular Motion')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Angular Velocity (Degrees / second')
        ax1.set_ylim(-40,40)
        
        if(len(rep_marker_locs)>0):
            ax2.plot(tvals, magvals, '-gD', markevery=rep_marker_locs,markersize=20)
        else:
            ax2.plot(tvals, magvals, '-g')
            ax2.plot(tvals[steps_indicies], magvals[steps_indicies], 'r*')
        ax2.set_title('Real Time Magnitude')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Angular Velocity (Degrees / second)')
        ax2.set_ylim(-40,40)
    except KeyboardInterrupt:
        quit()

try:
    #This socket is used to receive data from the data collection server
    receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receive_socket.connect(("none.cs.umass.edu", 8888))
    # ensures that after 1 second, a keyboard interrupt will close
    receive_socket.settimeout(1.0)
    
    print("Authenticating user for receiving data...")
    sys.stdout.flush()
    authenticate(receive_socket)
    
    print("Successfully connected to the server! Waiting for incoming data...")
    sys.stdout.flush()
        
    previous_json = ''

    t = 0
    x = 0
    y = 0
    z = 0
    
    #numpy array buffers used for visualization
    tvals = np.linspace(0,10,num=250)
    xvals = np.zeros(250)
    yvals = np.zeros(250)
    zvals = np.zeros(250)
    magvals = np.zeros(250)
    repindices = np.zeros(250,dtype='int')
    
    socketThread = threading.Thread(target=recv_data, args=())
    socketThread.start()
    
    #Setup the matplotlib plotting canvas
    style.use('fivethirtyeight')
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    # Point to the animation function above, show the plot canvas
    ani = animation.FuncAnimation(fig, animate, interval=20)
    plt.show()

except KeyboardInterrupt: 
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Quitting...")
    plt.close("all")
    quit()