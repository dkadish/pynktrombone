import time
import pastream as ps
import numpy as np

# A simple tone generator
def tone_generator(stream, buffer, f, loop=False):
    fs = stream.samplerate

    # Create a time index
    t = 2*np.pi*f*np.arange(len(buffer), dtype=stream.dtype) / fs
    i = 1.0

    # Loop until the stream stops
    while not stream.finished:
        frames = buffer.write_available
        print('Status: {}, Frames: {}, Framecount: {}, XRuns: {}'.format(stream.status, frames, stream.frame_count, stream.xruns))
        if not frames:
            time.sleep(0.010)
            continue

        buffer.write(np.sin(i*t))

        # # Get the write buffers directly to avoid making any extra copies
        # frames, part1, part2 = buffer.get_write_buffers(frames)
        #
        # out = np.frombuffer(part1, dtype=stream.dtype)
        # np.sin(i*t[:len(out)], out=out)
        #
        # if len(part2):
        #     # part2 will be nonempty whenever we wrap around the end of the ring buffer
        #     out = np.frombuffer(part2, dtype=stream.dtype)
        #     np.sin(i*t[:len(out)], out=out)
        #
        # # flag that we've added data to the buffer
        # buffer.advance_write_index(frames)

        # advance the time index
        t += 2*np.pi*f*frames / fs

        i/=1.0001

    print('Final Status: {}, Aborted ({}), Finished ({})'.format(stream.status, stream.aborted, stream.finished))

with ps.OutputStream(channels=1) as stream:
    # Set our tone generator as the source and pass along the frequency
    freq = 1000
    stream.set_source(tone_generator, args=(freq,))

    # Busy-wait to allow for keyboard interrupt
    stream.start()
    while stream.active:
        time.sleep(0.1)