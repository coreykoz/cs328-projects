import socket
import sys
import json
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.signal import find_peaks, butter, filtfilt

# TODO: Replace the string with your user ID
user_id = "grp404"

# TODO: (optional) Initialize any global variables you may need for your step detection algorithm
global filtered_ppg
global heart_indicies, ___

#################   Begin Server Connection Code  ####################

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
        raise Exception(
            "Expected message with prefix '{}' from server, received {}".format(msg_acknowledge_id, message))

    if (ack_id == user_id):
        print("Authentication successful.")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception(
            "Authentication failed : Expected user ID '{}' from server, received '{}'".format(user_id, ack_id))


def recv_data():
    """
    Continuously receives data from the server and calls detectSteps
    """
    global receive_socket
    global t, value  # global variables to hold incoming timestamp, x, y and z values
    global tvals, valueArr

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
                previous_json = ''  # reset if all were successful
                sensor_type = data['sensor_type']
                if (sensor_type == u"SENSOR_PPG"):
                    t = data['data']['t']
                    value = data['data']['value']

                    # Shift new data into the numpy plot buffers
                    valueArr = shift(valueArr, 1, cval=0)
                    valueArr[0] = value



            sys.stdout.flush()
            detectSteps(t, valueArr)
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


#################   End Server Connection Code  ####################

def detectSteps(t, valueArr):
    """
    Accelerometer-based step detection algorithm.

    In this assignment, you will implement a step detection algorithm for
    live accelerometer data collected from your Android app. This may be
    functionally equivalent to your step detection algorithm for static data
    if you like. Remember to use the global keyword if you would like to
    access global variables such as counters or buffers.
    """

    # TODO: Beat detection Algorithm
    global filtered_ppg
    global heart_indicies, ___
    # FILTEREING

    #High pass filtering/butterworth
    order = 5
    fs = 50.0  # sample rate, Hz
    cutoff = 7  # desired cutoff frequency of the filter, Hz. MODIFY AS APPROPROATE

    # Create the filter.
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    filtered_ppg = filtfilt(b, a, valueArr)

    heart_indicies, ___ = find_peaks(filtered_ppg, height=.10420)

    # HEART BEATS
    time = len(tvals)
    beats = len(heart_indicies)

    bpm = (beats * 6)
    print("Heart Rate over 5 seconds: " + str(bpm) + ' '+str(beats))

    return


def animate(i):
    """
    Helper function that animates the canvas
    """
    global tvals, valueArr  # global value buffers that we are appending to in recv_data
    global filtered_ppg
    global heart_indicies, ___

    try:
        ax1.clear()
        ax2.clear()

        # plotting live values of acceleration in the x, y and z directions
        ax1.plot(tvals, valueArr, label="values")
        ax1.legend(loc='upper right')
        ax1.set_title('Real Time PPG')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_ylim(150, 350)

        # TODO: add code to plot magnitude on axis 2. Also add markers to the plot at points where steps are detected.

        # boilerplate code for the layout of graph 2
        ax2.plot(tvals, filtered_ppg, label="values")
        ax2.plot(tvals[heart_indicies], filtered_ppg[heart_indicies], 'b*', label="heartbeats")
        ax2.set_title('Filtered PPG')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.set_ylim(-1, 1)

    except KeyboardInterrupt:
        quit()


try:
    # This socket is used to receive data from the data collection server
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

    # numpy array buffers used for visualization
    tvals = np.linspace(0, 10, num=250)
    valueArr = np.zeros(250)
    magvals = np.zeros(250)
    stepindices = np.zeros(250, dtype='int')

    socketThread = threading.Thread(target=recv_data, args=())
    socketThread.start()

    # Setup the matplotlib plotting canvas
    style.use('fivethirtyeight')

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Point to the animation function above, show the plot canvas
    ani = animation.FuncAnimation(fig, animate, interval=20)
    plt.show()

except KeyboardInterrupt:
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Quitting...")
    plt.close("all")
    quit()