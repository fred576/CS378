import numpy as np
import pyaudio
import struct
from itertools import combinations

G = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0]])
H = np.concatenate(( G[:, 20:].T, np.eye(10)), axis=1)
# n = H.shape[1]
# syndrome_map = {}
# for error_weight in range(1, 3):
#     for error_indices in combinations(range(n), error_weight):
#         error_vector = np.zeros(n, dtype=int)
#         error_vector[list(error_indices)] = 1
#         syndrome = tuple(np.mod(np.dot(H, error_vector), 2))
#         syndrome_map[syndrome] = error_vector

def encode(message, G):
    return np.mod(np.dot(message, G), 2)

# def decode(received, H, syndrome_map):
#     syndrome = tuple(np.mod(np.dot(received, H.T), 2))
#     error_vector = syndrome_map.get(syndrome, np.zeros(H.shape[1], dtype=int))
#     decoded = np.mod(received + error_vector, 2)
#     return decoded

def transmit_bitstring(bitstring, length, sample_rate=44100, duration=7.0):
    p = pyaudio.PyAudio()

    base_freq = 4000
    bin_size = 100
    num_bits = len(bitstring)

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    signal = np.zeros_like(t)

    for i in range(num_bits):
        bit = bitstring[i]
        if bit == 1:
            freq = base_freq + (i + 5.5) * bin_size
            signal += np.sin(2 * np.pi * freq * t)

    length_str = str(bin(length))[2:]
    length_str = '0'*(5 - len(length_str)) + length_str
    print(length_str)
    for i in range(len(length_str)):
        bit = int(length_str[i])
        if bit == 1:
            freq = base_freq + (i + 0.5) * bin_size
            signal += np.sin(2 * np.pi * freq * t)
    
    signal *= 32767 / np.max(np.abs(signal))
    signal = signal.astype(np.int16)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    for sample in signal:
        stream.write(struct.pack('h', sample))

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Bitstring transmitted successfully.")

bitstring = np.array([int(d) for d in input("Enter bitstring: ")])
p,q = tuple(map(float, input().split()))
print(f"Received codeword {bitstring}")

padded_bitstring = bitstring.tolist() + [0]*(G.shape[0] - len(bitstring))
message = encode(padded_bitstring,G)
print(f"Encoded message {message}")
e_1, e_2 = np.ceil(p*len(message)), np.ceil(q*len(message))
message[int(e_1)] ^= 1
message[int(e_2)] ^= 1
print(f"Corrupted message {message}")

transmit_bitstring(message, len(bitstring))