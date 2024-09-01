import numpy as np
import pyaudio
import struct
from itertools import combinations

# Generator matrix for a [20, 30, 5] code with 10 parity bits
# Picked from the internet, website detailed in the Design Doc
G = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0]])
# Parity check matrix for the same code
H = np.concatenate(( G[:, 20:].T, np.eye(10)), axis=1)

def encode(message, G):
    return np.mod(np.dot(message, G), 2)

def binary_array_to_string(binary_array):
    return ''.join(str(bit) for bit in binary_array)

def send_message(bitstring, freq_base=4000, bin_size=20, freq_high=8000, sample_rate=44100):
    #Length is the first 5 bits, rest 30 bits are the message - 20 data + 10 parity bits
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
    print("Message sent successfully")

bitstring = np.array([int(d) for d in input("Enter bitstring: ")])
p,q = tuple(map(float, input("Enter alpha and beta: ").split()))
print(f"Received codeword {bitstring}")

padded_bitstring = bitstring.tolist() + [0]*(G.shape[0] - len(bitstring))
message = encode(padded_bitstring,G)
print(f"Encoded message {message}")

e_1, e_2 = np.ceil(p*len(message)), np.ceil(q*len(message))
message[int(e_1)] ^= 1
message[int(e_2)] ^= 1
print(f"Corrupted message {message}")

padded_length = '0' * (5 - len(bin(len(bitstring))[2:])) + bin(len(bitstring))[2:]
print(padded_length)
print(binary_array_to_string(message))
send_message(padded_length + binary_array_to_string(message))
