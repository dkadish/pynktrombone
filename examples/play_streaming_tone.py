import numpy as np

import sounddevice as sd

sd.default.samplerate = 48000
sd.default.dtype = np.float32

duration = 2.0  # seconds

starting = 0


def process(outdata, frames, time, status):
    global starting, sd

    if status:
        print(status)
    j = starting + np.arange(frames, dtype=sd.default.dtype[1]) * 2.0 * np.pi * 400.0 / sd.default.samplerate
    wave = np.sin(j)
    outdata[:] = np.repeat(wave.reshape(-1, 1), 2, axis=1)
    starting = j[-1]


def main():
    with sd.OutputStream(channels=2, callback=process, blocksize=512, samplerate=sd.default.samplerate):
        sd.sleep(int(duration * 1000))


if __name__ == '__main__':
    main()
