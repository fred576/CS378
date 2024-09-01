import numpy as np
import pyaudio
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from itertools import combinations

#error correction
G = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0]])
H = np.concatenate(( G[:, 20:].T, np.eye(10)), axis=1)

n = H.shape[1]
syndrome_map = {}
for error_weight in range(1, 3):
    for error_indices in combinations(range(n), error_weight):
        error_vector = np.zeros(n, dtype=int)
        error_vector[list(error_indices)] = 1
        syndrome = tuple(np.mod(np.dot(H, error_vector), 2))
        syndrome_map[syndrome] = error_vector

#message encoding function
def encode(message, G):
    return np.mod(np.dot(message, G), 2)

#message decoding function
def decode(received, H, syndrome_map):
    received = np.array([int(x) for x in received])
    print(received)
    syndrome = tuple(np.mod(np.dot(received, H.T), 2))
    error_vector = syndrome_map.get(syndrome, np.zeros(H.shape[1], dtype=int))
    decoded = np.mod(received + error_vector, 2)
    return decoded


#peak_detection
def detect_peaks(x, num_bins):
    binsize = len(x)//num_bins
    bitstring = ""

    for i in range(num_bins):
        start = i * binsize
        end = start + binsize
        #current bin
        x_bin = x[start:end]

        #finding the maxima of the bin
        middle_start = binsize // 2 - binsize//10
        middle_end = binsize // 2 + binsize//10
        middle_range_y = x_bin[middle_start:middle_end]
    
        middle_max = np.max(middle_range_y)
        threshold = 0.55 * middle_max 

        left_range_y = x_bin[:middle_start-binsize//8]
        right_range_y = x_bin[middle_end+binsize//8:]

        #threshold checking and intensity filtering for peak detection (TO BE DONE MANUALLY based on amplitude values)
        if np.all(left_range_y < threshold) and np.all(right_range_y < threshold) and middle_max>1e5:
            bitstring += '1' 
        else:
            bitstring += '0'
    
    return bitstring    

def receive_bitstring_with_fft(sample_rate=44100, duration = 1.0):
    p = pyaudio.PyAudio()
    base_freq = 4000
    bin_size = 100 
    num_bits = 35   
    
    chunk_size = int(sample_rate * duration)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

     #listening for bitstring 
    print("Listening for bitstring...")
    data = stream.read(chunk_size)
    print("Received audio data.")

    #taking fft of the given frequency
    audio_signal = np.frombuffer(data, dtype=np.int16)
    freqs = np.fft.fftfreq(len(audio_signal), 1/sample_rate)
    fft_spectrum = np.abs(np.fft.fft(audio_signal))
    positive_freqs = freqs[:len(freqs)//2]
    positive_spectrum = fft_spectrum[:len(fft_spectrum)//2]
    
    #clipping the frequency range
    clip_start = np.searchsorted(positive_freqs, base_freq)
    clip_end = np.searchsorted(positive_freqs, base_freq + num_bits * bin_size)
    clipped_freqs = positive_freqs[clip_start:clip_end]
    clipped_spectrum = positive_spectrum[clip_start:clip_end]
    print(len(clipped_spectrum))
    
    #fft plot
    plt.plot(clipped_freqs, clipped_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Clipped FFT Spectrum')
    plt.grid(True)

    bitstring = detect_peaks(clipped_spectrum, num_bits)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print(f"Received bitstring: {bitstring}")
    return bitstring


received = receive_bitstring_with_fft()
length = int(received[:5], 2)
print(length)
decoded = decode(received[5:], H, syndrome_map)

print(f"Decoded codeword : {decoded}")
print(f"Actual hopefuully {decoded[:length]}")
plt.show()
