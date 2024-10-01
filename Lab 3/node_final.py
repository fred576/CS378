import numpy as np
import pyaudio
import struct
from statistics import mode
from itertools import combinations
import matplotlib.pyplot as plt
import threading
import datetime

node_id = None #enter node id here
all_messages = []
all_destinations = []
universal_rts = [{}, {2: 7200, 3: 7600}, {1:7200, 3: 6800}, {1: 7600, 2: 6800}]
universal_cts = [{}, {2: 7400, 3: 7800}, {1:7400, 3: 7000}, {1: 7800, 2: 7000}]
rts = universal_rts[node_id]
cts = universal_cts[node_id]

# Generator matrix for [30, 20, 5] code with 10 paritty bits and min. distance between codewords 5
# Picked from the internet - website detailed in design doc
G = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0]])
# Parity check matrix for the same code, such that GH^T = 0
H = np.concatenate(( G[:, 20:].T, np.eye(10)), axis=1)

n = H.shape[1]
# Using syndrome Decoding. All codewords are generated using linear combination of rows of G.
# Thus if received codeword c* = c + e, where e is error bit with max bits 2
# Hc* = Hc + He = 0 + He = He
# thus by mapping He to e for each valid e with max bits 2, it is possible to know what e is, and then get c = c* + e as the true answer
syndrome_map = {}
for error_weight in range(1, 3):
    for error_indices in combinations(range(n), error_weight):
        error_vector = np.zeros(n, dtype=int)
        error_vector[list(error_indices)] = 1
        syndrome = tuple(np.mod(np.dot(H, error_vector), 2))
        syndrome_map[syndrome] = error_vector
def decode(received, H, syndrome_map):
    received = np.array([int(x) for x in received])
    # print(received)
    syndrome = tuple(np.mod(np.dot(received, H.T), 2))
    error_vector = syndrome_map.get(syndrome, np.zeros(H.shape[1], dtype=int))
    decoded = np.mod(received + error_vector, 2)
    return decoded
def encode(message, G):
    return np.mod(np.dot(message, G), 2)
def convert_new(l, freq_high):
    i = 0
    actual = []
    ans = []
    temp = []
    # This function is used to convert the dominant frequency at each sampling point to an interval
    for x in l:
        if abs(x-freq_high)<1000:
            if temp:
                ans.append(temp)
                temp = []
        else:
            temp.append(x)
    if temp:
        ans.append(temp)
    # This part is used to remove anomalies that may occur in the transmitted frequencies
    for l in ans:
        if l.count(mode(l))>2:
            actual.append(mode(l))
    return actual

def RTS_sender(freqs, duration, sample_rate=44100):
    print(f"sending freqs: {freqs}")
    p = pyaudio.PyAudio()
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    signal = np.zeros_like(t)
    for freq in freqs:
        signal += np.sin(2 * np.pi * freq * t)
    signal *= 32767 / np.max(np.abs(signal))
    signal = signal.astype(np.int16)
    # signal = np.int16(signal * 32767)
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True)
    for sample in signal:
        stream.write(struct.pack('h', sample))
    stream.stop_stream()
    stream.close()
    p.terminate()

def send_RTS(d, duration = 1):
    print("Sending RTS to destination ", d)
    if(d == 0):
        RTS_sender([rts[i] for i in range(1, 4) if i!= node_id],duration)
    else:
        RTS_sender([rts[d]],duration)
    return

def CTS_detector(freqs_to_detect, duration, sample_rate=44100):
    p = pyaudio.PyAudio()
    chunk_duration = 0.05  # Check every 0.05 seconds
    chunk_size = int(sample_rate * chunk_duration)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    # print("Listening for message...")
    l = []
    # received_bits = ''
    
    for _ in range(int(duration / chunk_duration)):
        data = stream.read(chunk_size)
        audio_signal = np.frombuffer(data, dtype=np.int16)

        # FFT to determine the dominant frequency
        freqs = np.fft.fftfreq(len(audio_signal), 1/sample_rate)
        fft_spectrum = np.fft.fft(audio_signal)

        total_fft = fft_spectrum
        total_fft[freqs < 3800] = 0
        total_fft[freqs > 8200] = 0
        total_fft[abs(freqs - universal_rts[node_id%3 + 1][(node_id+1)%3 + 1]) < 100] = 0
        total_fft[abs(freqs - universal_cts[node_id%3 + 1][(node_id+1)%3 + 1]) < 100] = 0

        for freq in freqs_to_detect:
            # clipped_fft_spectrum = fft_spectrum
            # clipped_fft_spectrum[freqs < freq- 50] = 0
            # clipped_fft_spectrum[freqs > freq + 50] = 0
            clipped_fft_spectrum = np.copy(total_fft)
            for i in range(len(freqs_to_detect)):
                if(freqs_to_detect[i] == freq):
                    continue
                else:
                    clipped_fft_spectrum[abs(freqs - freqs_to_detect[i]) < 100] = 0
            dominant_freq = abs(freqs[np.argmax(np.abs(clipped_fft_spectrum))])
            average_noise = np.mean(np.abs(clipped_fft_spectrum))
            peak = np.max(np.abs(clipped_fft_spectrum))
            # print("dominant freq" , dominant_freq)
            # print("peak : ", peak)
            # print("average noise : ", average_noise)
            
            if(np.abs(dominant_freq - freq) < 25 and peak/average_noise > 30 and peak > 100000):
                l += [1]
            else:
                l += [0]
        

        # plt.plot(freqs, np.abs(total_fft))
        # plt.xlim(3800, 8200)
        # plt.xlabel('Frequency')
        # plt.ylabel('Amplitude')
        # plt.title('FFT Spectrum')
        # plt.savefig(f'total_fft.png')
        # plt.close()
        peak = np.max(np.abs(total_fft))
        average_noise = np.mean(np.abs(total_fft))
    stream.stop_stream()
    stream.close()
    p.terminate()

    freq_detected = []
    for i in range(len(freqs_to_detect)):
        sub_entries = l[i::len(freqs_to_detect)]
        mode_entry = np.sum(sub_entries)
        freq_detected.append(1 if (mode_entry>=4) else 0)
    
    # print("Peak: ", peak)
    # print("Average noise: ", average_noise)
    # print("freq_detected: ", freq_detected)

    # print(l)
    return freq_detected, (peak > 10000 and peak/average_noise > 100)
    
def check_CTS(d, duration = 2):
    print("Checking CTS from destination ", d)
    f=[]
    if(d == 0):
        f=CTS_detector([cts[d] for d in range(1, 4) if d!= node_id],duration)[0]
    else:
        f=CTS_detector([cts[d]],duration)[0]
    return [1]*len(f)==f

def send_message(message, freq_base=4000, bin_size=20, freq_high=8000, sample_rate=44100):
    #Length is the first 5 bits, rest 30 bits are the message - 20 data + 10 parity bits
    # print(f"sending message : {message}")
    message_array = np.array([int(d) for d in message])
    padded_message = message_array.tolist() + [0]*(G.shape[0] - len(message))
    encoded_message = encode(padded_message,G)
    padded_length = '0' * (5 - len(bin(len(message))[2:])) + bin(len(message))[2:]
    bitstring = padded_length + ''.join([str(x) for x in encoded_message])
    
    assert len(bitstring) == 35, "Bitstring must be 35 bits long"
    
    p = pyaudio.PyAudio()
    duration = 0.3  
    chunk_size = 7  

    chunks = [bitstring[i:i+chunk_size] for i in range(0, len(bitstring), chunk_size)]

    for i, chunk in enumerate(chunks):
        value = int(chunk, 2)  # Convert chunk to an integer
        freq = freq_base + bin_size * value  # Calculate frequency for this chunk

        print(freq)

        if(i == 0):
            t = np.linspace(0, 8* duration, int(sample_rate * 8 * duration), False)
        else:
            t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.sin(2 * np.pi * freq * t)
        signal = np.int16(signal * 32767)  # Normalize to 16-bit range
        
        # Play the signal
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        output=True)
        for sample in signal:
            stream.write(struct.pack('h', sample))
        stream.stop_stream()
        stream.close()

        # Send high frequency as separator
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        separator_signal = np.sin(2 * np.pi * freq_high * t)
        separator_signal = np.int16(separator_signal * 32767)
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        output=True)
        for sample in separator_signal:
            stream.write(struct.pack('h', sample))
        stream.stop_stream()
        stream.close()

    p.terminate()
    # print("Message sent successfully")

def listen_and_receive(freq_base=4000, bin_size=20, freq_high=8000, sample_rate=44100, duration=10.0):
    p = pyaudio.PyAudio()
    chunk_duration = 0.05  # Check every 0.05 seconds
    chunk_size = int(sample_rate * chunk_duration)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Listening for message...")
    l = []
    received_bits = ''
    
    for _ in range(int(duration / chunk_duration)):
        data = stream.read(chunk_size)
        audio_signal = np.frombuffer(data, dtype=np.int16)

        # FFT to determine the dominant frequency
        freqs = np.fft.fftfreq(len(audio_signal), 1/sample_rate)
        fft_spectrum = np.fft.fft(audio_signal)
        clip_base = freq_base - 500
        #remove the lower frequencies
        fft_spectrum[freqs< clip_base] = 0
        dominant_freq = abs(freqs[np.argmax(np.abs(fft_spectrum))])
        # print(dominant_freq)
        l.append(dominant_freq)
    listof = convert_new(l, freq_high)
    print(listof)
    for x in listof[:5]:
        freq = int((x-freq_base)/bin_size)
        if(freq>=0):
            received_bits += '0'*(7 - len(str(bin(freq)[2:]))) + str(bin(freq)[2:])

    stream.stop_stream()
    stream.close()
    p.terminate()


    if len(received_bits) != 35:
        print(f"Warning: Received bitstring length ({len(received_bits)}) does not match expected length (35)")
    
    print(f"Received 35-bit message: {received_bits}")
    return received_bits

def gotoreceive(duration, freq_base=4000, bin_size=20, freq_high=8000, sample_rate=44100):
    # you listen and if in some chunk you hear your frequency
    # then t+=sometime and send cts
    # if t=0 then return only if last one second nothing was heard
    chunk_duration = 0.5  # Check every 0.05 seconds
    lastseen=0.00
    i=0
    print("duration/chunk_duration",duration/chunk_duration)
    while(i<duration/chunk_duration):
        i+=1
        print(i)
        x,y=CTS_detector([rts[d] for d in range(1, 4) if d!= node_id],1) 
        if(y):
            lastseen=i*chunk_duration
        if(x==[0]*len(x)):
            continue
        fr = 0
        index = 0
        if(x[0]):
            # duration+=5
            index = [i for i in range(1,4) if i != node_id][0]
            fr = cts[index]
            RTS_sender([fr],2) #cts hai
            
        elif(x[1]):
            # duration+=5
            index = [i for i in range(1,4) if i != node_id][1]
            fr = cts[index]
            RTS_sender([fr],2) #cts hai
        received = listen_and_receive()
        if(len(received) == 35):
            decoded = decode(received[5:], H, syndrome_map)
            length = int(received[:5], 2)
            # print(length)
            # print(f"Decoded codeword : {decoded}")
            # print(f"Actual hopefuully {decoded[:length]}")
            time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[RECVD] {decoded[:length]} from NODE ID: {index} at {time}")
        else:
            print("Error in receiving, going back to loop")
    # print(lastseen)
    if(lastseen>duration-2):
        gotoreceive(np.random.rand()*4+2.1)
    # return 1

messages = []
destinations = []

def sending_and_receiving():
    while True:
        if(len(all_messages) == 0 and len(messages) == 0):
            gotoreceive(1000000)
        elif(len(messages) == 0):
            gotoreceive(5)
        else:
            bitstring = messages[0]
            destination = destinations[0]
            gotoreceive(np.random.rand()*4 + 2.1)
            send_RTS(destination)
            if(check_CTS(destination)==0):
                continue
            send_message(bitstring)
            time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[SENT] {bitstring} to NODE ID: {destination} at {time}")
            messages.pop(0)
            destinations.pop(0)

def print_on_enter():
    while True:
        input()  # Wait for the Enter key
        if len(all_messages) > 0:
            messages.append(all_messages[0])
            destinations.append(int(all_destinations[0]))
            all_messages.pop(0)
            all_destinations.pop(0)
        
m1, dest1 = input().split()
m2, dest2 = input().split()
if(dest1 != -1):
    all_messages.append(m1)
    all_destinations.append(dest1)
if(dest2 != -1):
    all_messages.append(m2)
    all_destinations.append(dest2)
all_messages = [m1, m2]
all_destinations = [dest1, dest2]

task_thread = threading.Thread(target=sending_and_receiving)
task_thread.daemon = True
task_thread.start()

# Start the Enter key detection in the main thread
print_on_enter()