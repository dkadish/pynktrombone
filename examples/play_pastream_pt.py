import time
import pastream as ps
import numpy as np

from voc import Voc, CHUNK

# A simple tone generator
def tone_generator(stream, buffer, loop=False):
    fs = stream.samplerate

    vocal = Voc(fs)

    # Loop until the stream stops
    while not stream.finished:
        frames = buffer.write_available
        if frames < CHUNK:
            time.sleep(0.010)
            continue

        # Get the write buffers directly to avoid making any extra copies
        frames, part1, part2 = buffer.get_write_buffers(frames)

        # Calculate vocal data
        _osc = 0
        _voc = 0
        if vocal.counter == 0:
            _osc = 12 + 16 * (0.5 * (_osc + 1))
            vocal.tongue_shape(_osc, 2.9)

        out = np.array(vocal.compute(use_np=True), dtype=np.float32)[:CHUNK].reshape(-1)

        outbuff = np.frombuffer(part1, dtype=stream.dtype)
        first_buff_len = len(outbuff)
        if first_buff_len < CHUNK:
            outbuff[:] = out[:first_buff_len]
            if len(part2):
                # part2 will be nonempty whenever we wrap around the end of the ring buffer
                outbuff = np.frombuffer(part2, dtype=stream.dtype)
                outbuff[:(CHUNK-first_buff_len)] = out[first_buff_len:]
        else:
            outbuff[:CHUNK] = out

        # flag that we've added data to the buffer
        buffer.advance_write_index(CHUNK)
        print('Status: {}, Frames: {}, Framecount: {}, XRuns: {}, CPU: {}'.format(stream.status, frames, stream.frame_count, stream.xruns, stream.cpu_load))

    print('Final Status: {}, Aborted ({}), Finished ({})'.format(stream.status, stream.aborted, stream.finished))

with ps.OutputStream(channels=1) as stream:
    # Set our tone generator as the source and pass along the frequency
    stream.set_source(tone_generator, args=())

    # Busy-wait to allow for keyboard interrupt
    stream.start(prebuffer=True)
    while stream.active:
        time.sleep(0.1)