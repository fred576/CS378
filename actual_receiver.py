import numpy as np
import pyaudio
from scipy.signal import find_peaks
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from itertools import combinations

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

def encode(message, G):
    return np.mod(np.dot(message, G), 2)

def decode(received, H, syndrome_map):
    received = np.array([int(x) for x in received])
    print(received)
    syndrome = tuple(np.mod(np.dot(received, H.T), 2))
    error_vector = syndrome_map.get(syndrome, np.zeros(H.shape[1], dtype=int))
    decoded = np.mod(received + error_vector, 2)
    return decoded

# Function to detect peaks and return the bitstring
def detect_peaks(x, num_bins):
    binsize = len(x)//num_bins
    bitstring = ""

    for i in range(num_bins):
        # Get the bin range
        start = i * binsize
        end = start + binsize
        
        # Extract the current bin
        x_bin = x[start:end]
        
        # Define the range around the middle for finding the peak
        middle_start = binsize // 2 - binsize//20
        middle_end = binsize // 2 + binsize//20
        middle_range_y = x_bin[middle_start:middle_end]
        
        # Get the max value in the middle range
        middle_max = np.max(middle_range_y)
        threshold = 0.7 * middle_max  # 20% of the middle max
        
        # Define the left and right ranges outside the middle
        left_range_y = x_bin[:middle_start-binsize//10]
        right_range_y = x_bin[middle_end+binsize//10:]
        
        # Check if all values in left and right ranges are below the threshold
        if np.all(left_range_y < threshold) and np.all(right_range_y < threshold):
            bitstring += '1'  # Peak detected
        else:
            bitstring += '0'  # No distinct peak detected
    
    return bitstring

def receive_bitstring_with_fft(sample_rate=44100, duration = 2.0):
    p = pyaudio.PyAudio()

    # Parameters
    base_freq = 4000  # Starting frequency in Hz
    bin_size = 100    # Size of each frequency bin in Hz
    num_bits = 35     # Number of bits to decode
    
    chunk_size = int(sample_rate * duration)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Listening for bitstring...")

    data = stream.read(chunk_size)

    print("Received audio data.")
    audio_signal = np.frombuffer(data, dtype=np.int16)

    # output_file = 'received_audio.wav'
    # wav.write(output_file, sample_rate, audio_signal)
    # print(f"Audio signal saved as {output_file}")
    # print(f"Audio signal saved as {output_file}")
    # Perform FFT on the received signal
    freqs = np.fft.fftfreq(len(audio_signal), 1/sample_rate)
    fft_spectrum = np.abs(np.fft.fft(audio_signal))

    # Consider only the positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_spectrum = fft_spectrum[:len(fft_spectrum)//2]

    # Clip frequencies below 7000 and above 10000
    clip_start = np.searchsorted(positive_freqs, base_freq)
    clip_end = np.searchsorted(positive_freqs, base_freq + num_bits * bin_size)
    clipped_freqs = positive_freqs[clip_start:clip_end]
    clipped_spectrum = positive_spectrum[clip_start:clip_end]

    # Plot clipped frequency vs amplitude
    
    # plt.plot(clipped_freqs, clipped_spectrum)
    # #plt.plot(positive_freqs, positive_spectrum)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.title('Clipped FFT Spectrum')
    # plt.grid(True)

    # Add dotted vertical lines
    # for freq in range(base_freq, base_freq + bin_size * num_bits, bin_size):
    #     plt.axvline(x=freq, linestyle='dotted', color='red')

    # Save the plot as a PNG file
    # plt.savefig('freqs.png')
    # plt.close()
    # Use peak detection to find significant frequencies
    # peaks, _ = find_peaks(positive_spectrum, height=1000)  # Adjust height as needed
    # peak_freqs = positive_freqs[peaks]

    # Decode the bitstring based on the presence of peaks in the frequency bins
    # bitstring = ''

    # start = base_freq
    bitstring = detect_peaks(clipped_spectrum, num_bits)
    # threshold = 5
    # bin_values_list = []
    # for i in range(num_bits):
    #     bin_values_list.append([])
    # for value in range(len(clipped_freqs)):
    #     bin_values_list[(int(clipped_freqs[value]-start))//bin_size].append(clipped_spectrum[value])
    # normalised_freq = np.arange(0,50,1)
    # for i in range(num_bits):
    #     bin_start = start + i * bin_size
    #     bin_end = bin_start + bin_size
        
    #     bin_values = bin_values_list[i]
    #     plt.plot(bin_values)
    # #     #print(type(bin_values[0]))
    #     plt.savefig(f'test_{i}.png')
    #     plt.close()
    #     if bin_values:
    #         # Calculate the peak (maximum value) and variance
    #         peak = max(bin_values)
    #         #variance = np.sqrt(np.var(bin_values))
    #         mean = np.mean(bin_values)
    #         print(peak/mean)
    #         if (peak /mean) > threshold:
    #             bitstring += '1'
    #         else:
    #             bitstring += '0'
    #     else:
    #         bitstring += '0'

    stream.stop_stream()
    stream.close()
    p.terminate()
    #length = int(bitstring[:5], 2)
    
    print(f"Received bitstring: {bitstring}")
    return bitstring

# Example usage
received = receive_bitstring_with_fft()
length = int(received[:5], 2)
decoded = decode(received[5:], H, syndrome_map)

print(f"Decoded codeword : {decoded}")
print(f"Actual hopefuully {decoded[:length]}")
