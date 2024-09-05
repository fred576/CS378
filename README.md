# CS378_Lab2

This project implements an audio communication system using syndrome decoding for error correction. The system transmits data encoded in the form of audio frequencies, and the receiver decodes the message using error-correcting codes. The project uses a [30, 20, 5] linear block code with 30-bit codewords and 20 information bits, ensuring error detection and correction up to 2 bits of error.

## Project Structure

- **`actual_sender.py`**: Script responsible for encoding and sending data as audio signals.
- **`actual_receiver.py`**: Script responsible for listening to audio signals, detecting dominant frequencies, and decoding the received message.
- **`README.md`**: Project documentation.

## Features

- **Error Correction**: Uses syndrome decoding based on a [30, 20, 5] linear block code.
- **Audio Transmission**: Transmits binary data using audio frequencies.
- **Frequency Detection**: Uses Fast Fourier Transform (FFT) to detect dominant frequencies in the received audio signal.

## Installation

### Prerequisites

- Python 3.x
- `pyaudio`: For capturing audio signals.
- `numpy`: For numerical computations (FFT, matrix operations, etc.).
- `statistics`: For handling statistical operations like mode detection.

Install the required Python libraries using `pip`:

```bash
pip install numpy pyaudio
