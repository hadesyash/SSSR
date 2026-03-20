import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Configuration ---
PORT = '/dev/tty.usbmodemF412FA6987F82'         
BAUD = 115200         
PAYLOAD_SIZE = 256    
MAGIC_HEADER = b'\xde\xad'
WINDOW_SAMPLES = 500  

# --- Setup ---
ser = serial.Serial(PORT, BAUD, timeout=0.1)
ser.reset_input_buffer()

recording = False
data_buffer = np.zeros((WINDOW_SAMPLES, 4))
csv_file = None
file_counter = 1

fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Arduino ADC Stream')
lines = [ax.plot(data_buffer[:, i], label=f'Ch {i}')[0] for i in range(4)]
ax.set_ylim(5000, 20000)  
ax.set_xlim(0, WINDOW_SAMPLES)
ax.legend(loc='upper right')
ax.set_title("Press SPACE to toggle, ESC to exit")

def toggle_recording():
    global recording, csv_file, file_counter
    recording = not recording
    if recording:
        ser.reset_input_buffer()
        ser.write(b'S')
        filename = f"{file_counter}.csv"
        csv_file = open(filename, 'w')
        csv_file.write("Ch0,Ch1,Ch2,Ch3\n")
        print(f"Recording STARTED. Saving to {filename}...")
    else:
        ser.write(b'E')
        if csv_file:
            csv_file.close()
        print(f"Recording STOPPED. Saved {file_counter}.csv.")
        file_counter += 1

def on_key(event):
    if event.key == ' ':
        toggle_recording()
    elif event.key == 'escape':
        print("\nExiting...")
        if recording:
            toggle_recording() 
        plt.close()

fig.canvas.mpl_connect('key_press_event', on_key)

def find_header():
    """Find magic header byte-by-byte"""
    while ser.in_waiting > 0:
        byte1 = ser.read(1)
        if byte1 == b'\xde':
            if ser.in_waiting > 0:
                byte2 = ser.read(1)
                if byte2 == b'\xad':
                    return True
        # Otherwise keep searching
    return False

def update(frame):
    global data_buffer, csv_file
    
    while recording:
        if ser.in_waiting >= (PAYLOAD_SIZE + 2):
            if find_header():  # Byte-by-byte search for header
                payload = ser.read(PAYLOAD_SIZE)
                
                if len(payload) == PAYLOAD_SIZE:
                    new_data = np.frombuffer(payload, dtype='<u2').reshape(-1, 4)
                    
                    if csv_file:
                        np.savetxt(csv_file, new_data, fmt='%d', delimiter=',')
                    
                    shift = new_data.shape[0]
                    data_buffer = np.roll(data_buffer, -shift, axis=0)
                    data_buffer[-shift:, :] = new_data
            else:
                break  # No complete frame available
        else:
            break  # Not enough data yet
    
    for i, line in enumerate(lines):
        line.set_ydata(data_buffer[:, i])
    return lines

print("Ensure the graph window is selected/focused to use keyboard controls.")

ani = FuncAnimation(fig, update, interval=10, blit=True, cache_frame_data=False)
plt.show()

# Cleanup
if recording and csv_file:
    csv_file.close()
ser.close()
print("Port closed.")