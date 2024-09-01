import numpy as np
import pyaudio
from statistics import mode
from itertools import combinations

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
    print(received)
    syndrome = tuple(np.mod(np.dot(received, H.T), 2))
    error_vector = syndrome_map.get(syndrome, np.zeros(H.shape[1], dtype=int))
    decoded = np.mod(received + error_vector, 2)
    return decoded

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
        received_bits += '0'*(7 - len(str(bin(freq)[2:]))) + str(bin(freq)[2:])

    stream.stop_stream()
    stream.close()
    p.terminate()


    if len(received_bits) != 35:
        print(f"Warning: Received bitstring length ({len(received_bits)}) does not match expected length (35)")
    
    print(f"Received 35-bit message: {received_bits}")
    return received_bits

# Example usage
received = listen_and_receive()
decoded = decode(received[5:], H, syndrome_map)
length = int(received[:5], 2)
print(length)
print(f"Decoded codeword : {decoded}")
print(f"Actual hopefuully {decoded[:length]}")