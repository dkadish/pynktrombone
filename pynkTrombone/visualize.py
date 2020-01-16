import logging
import os
import subprocess

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import neat
import numpy as np
from matplotlib.patches import Ellipse
from neat import Checkpointer
from scipy.io import wavfile

from evolution.player import PynkTrombonePlayer

# plt.style.use('dark_background')

import sys
sys.path.append('/Users/davk/Documents/phd/projects/ecosystem-soundscape/src/')

class VocalAnimator:

    def __init__(self, player, interval=0.25, filename='video'):
        # self.fig = plt.figure()
        # self.ax = plt.axes(xlim=(-1, 45), ylim=(-1, 5))

        self.fig, (self.ax_tract, self.ax_control) = plt.subplots(2, 1, figsize=(6.4,6.4), gridspec_kw={'height_ratios': [2, 1]})
        self.ax_tract.set_xlim(-1, 45)
        self.ax_tract.set_ylim(0, 3.5)

        # self.rest_line, = self.ax_tract.plot([], [], lw=1, color='black')
        self.current_line, = self.ax_tract.plot([], [], lw=2)
        self.target_line, = self.ax_tract.plot([], [], lw=2, linestyle='dashed')

        # self.rest = player.resting
        self.current = player.current
        self.target = player.target
        self.control = player.control
        self.interval = interval
        self.filename = filename

        self.slow_frame = -1

        self.control_tongue = Ellipse((0, 0), width=3, height=0.5, color='red')
        # self.ax_tract.add_collection(PatchCollection([self.control_tongue]))
        self.ax_tract.add_artist(self.control_tongue)
        self.control_bars = self.ax_control.bar(np.arange(len(self.control[0])), self.control[0])
        self.ax_control.set_ylim(-1.1, 1.1)

        # self.ax_control_lines.plot(self.control)

        # setting a title for the plot
        plt.title('PynkTrombone Output')
        # hiding the axis details
        # plt.axis('off')

    def _init_tract(self):
        # creating an empty plot/frame
        # self.rest_line.set_data([], [])
        self.current_line.set_data([], [])
        self.target_line.set_data([], [])
        # return [self.rest_line, self.current_line, self.target_line]
        return [self.current_line, self.target_line]

    def _init_control(self):
        # self.control_bars.set_data([], [])
        self.ax_control.set_xticks(np.arange(len(self.control[0])))
        self.ax_control.set_xticklabels(
            ('touch', 'freq', 'tense', 'tng_idx', 'tng_diam', 'velum', 'lips', 'epig', 'trach')
        )
        return [bar for bar in self.control_bars] + [self.control_tongue]

    def init(self):
        animators = self._init_tract()
        animators.extend(self._init_control())

        return animators

    def animate(self):
        # call the animator
        # frames = int(5.0/self.interval)
        frames = len(self.current)
        # interval = self.interval*1000.0
        interval = 5000.0/frames

        anim = animation.FuncAnimation(self.fig, self.frame, init_func=self.init,
                                       frames=frames, interval=interval, blit=True)

        # save the animation as mp4 video file
        anim.save('{}.mp4'.format(self.filename), writer='ffmpeg')

        # plt.show()

    def _frame_current_tract(self, i):
        xdata = np.arange(self.current[i].size)
        self.current_line.set_data(xdata, self.current[i])

        return [self.current_line]

    def _frame_tract(self, i):
        xdata = np.arange(self.current[i].size)
        # self.rest_line.set_data(xdata, self.rest)
        # self.current_line.set_data(xdata, self.current[i])
        self.target_line.set_data(xdata, self.target[i])
        # return [self.rest_line, self.current_line, self.target_line]
        # return [self.current_line, self.target_line]
        return [self.target_line]

    def _frame_control(self, i):
        for j, bar in enumerate(self.control_bars):
            bar.set_height(self.control[i][j])
        self.control_tongue.center = (32.0-10.0)*(self.control[i][3] + 1.0)/2.0 + 10.0, (.5-2.0)*(self.control[i][4] + 1.0)/2.0 + 2.0

        return [bar for bar in self.control_bars] + [self.control_tongue]

    def frame(self, i):
        animators = self._frame_current_tract(i)
        slow_frames = int(5.0/self.interval)
        slow_frame_interval = int(len(self.current) / slow_frames)

        if i > slow_frame_interval * (self.slow_frame + 1):
            self.slow_frame += 1
            try:
                animators.extend(self._frame_tract(self.slow_frame))
                animators.extend(self._frame_control(self.slow_frame))
            except IndexError as e:
                print('On frame {} of {}, slow frame {} is out of range.'.format(i, len(self.current), self.slow_frame))

        plt.tight_layout()

        return animators

def main(args):
    p = Checkpointer.restore_checkpoint(args.checkpoint)

    best_genome = p.best_genome
    if best_genome is None:
        evaluated = filter(lambda p: p.fitness is not None, p.population.values())
        by_fitness = sorted(evaluated, key=lambda p: p.fitness)
        best_genome = by_fitness[-1]

    nn = neat.nn.RecurrentNetwork.create(best_genome, p.config)
    player = PynkTrombonePlayer(nn, record=True)
    song = player.play(duration=5.0)

    wavfile.write('{}.wav'.format(args.filename), player.sample_rate, song)

    animator = VocalAnimator(player, filename=args.filename)
    animator.animate()

    f = os.path.abspath(args.filename)
    subprocess.run([
        'ffmpeg','-y',
        '-i','{}.mp4'.format(f),
        '-i','{}.wav'.format(f),
        '-map','0:v','-map','1:a','-c','copy',
        '{}.mov'.format(f)
    ])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize the vocalization')
    parser.add_argument('checkpoint', action='store')
    parser.add_argument('--filename', action='store', default='trombone')

    # parser.add_argument('features', action='store', default='./features.jbl')
    # parser.add_argument('config.conf', action='store', default='./config.conf')
    # parser.add_argument('-i', '--interval',  default=5)
    # parser.add_argument('--ignore-existing', action='store_true', default=False)
    # parser.add_argument('-b', '--bins', default=64)
    # parser.add_argument('--no-rois', action='store_true', default=False)
    # parser.add_argument('-k', '--keep-originals', default=False)

    # parser.add_argument('-e', '--extension', default='wav')
    #
    # parser.add_argument('-f', '--format', default=FORMAT)
    # parser.add_argument('-c', '--channels', default=CHANNELS)
    # parser.add_argument('-i', '--input', default=True)
    # parser.add_argument('-t', '--time', default=RECORD_SECONDS)

    args = parser.parse_args()
    print(args)
    logging.getLogger().setLevel(logging.WARN)
    print(logging.getLogger().getEffectiveLevel())

    main(args)