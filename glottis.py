import math
from random import random

import numpy as np

from tools import move_towards, sp_data

CHUNK = 512

class Glottis:
    def __init__(self, sr: float):
        self.enable = 1
        self.freq = 140
        self.tenseness = 0.6
        self.intensity = 0
        self.attack_time = 0.09
        self.release_time = 0.23
        self.time_in_waveform = 0

        self.Rd = None
        self.waveform_length = None

        self.alpha = None
        self.E0 = None
        self.epsilon = None
        self.shift = None
        self.delta = None
        self.Te = None
        self.omega = None

        self.setup_waveform()

        self.T = 1.0 / sr

        self.values = []

    #TODO Refactor for list of indices
    def setup_waveform(self):

        '''
        SPFLOAT
        Rd
        SPFLOAT
        Ra
        SPFLOAT
        Rk
        SPFLOAT
        Rg

        SPFLOAT
        Ta
        SPFLOAT
        Tp
        SPFLOAT
        Te

        SPFLOAT
        epsilon
        SPFLOAT
        shift
        SPFLOAT
        delta
        SPFLOAT
        rhs_integral

        SPFLOAT
        lower_integral
        SPFLOAT
        upper_integral

        SPFLOAT
        omega
        SPFLOAT
        s
        SPFLOAT
        y
        SPFLOAT
        z

        SPFLOAT
        alpha
        SPFLOAT
        E0
        '''


        # Derive Waveform length and Rd
        self.Rd = 3 * (1 - self.tenseness)
        self.waveform_length = 1.0 / self.freq

        Rd = self.Rd
        if Rd < 0.5:
            Rd = 0.5
        elif Rd > 2.7:
            Rd = 2.7

        # Derive Ra, Rk, and Rg
        Ra = -0.01 + 0.048 * Rd
        Rk = 0.224 + 0.118 * Rd
        Rg = (Rk / 4) * (0.5 + 1.2 * Rk) / (0.11 * Rd - Ra * (0.5 + 1.2 * Rk))

        # Derive Ta, Tp, and Te
        Ta = Ra
        Tp = 1.0 / (2.0 * Rg)
        Te = Tp + Tp * Rk

        # Calculate epsilon, shift, and delta
        epsilon = 1.0 / Ta
        shift = math.exp(-epsilon * (1.0 - Te))
        delta = 1 - shift

        # Calculate integrals
        rhs_integral = (1.0 / epsilon) * (shift - 1.0) + (1.0 - Te) * shift
        rhs_integral = rhs_integral / delta
        lower_integral = -(Te - Tp) / 2 + rhs_integral
        upper_integral = -lower_integral

        # Calculate E0
        omega = math.pi / Tp
        s = math.sin(omega * Te)

        y = -math.pi * s * upper_integral / (Tp * 2)
        z = math.log(y)
        alpha = z / (Tp / 2 - Te)
        E0 = -1 / (s * math.exp(alpha * Te))

        # Update Variables in glottis data structure
        self.alpha = alpha
        self.E0 = E0
        self.epsilon = epsilon
        self.shift = shift
        self.delta = delta
        self.Te = Te
        self.omega = omega

    def update(self, block_time: float):
        target_intensity = self.enable and 1 or 0
        self.intensity = move_towards(self.intensity,
                                      target_intensity,
                                      block_time / self.attack_time,
                                      block_time / self.release_time)

    def compute_np(self, randomize: bool = True) -> np.array:
        time_in_waveform = np.arange(self.T, self.T * (CHUNK + 1), step=self.T, dtype=np.float32)
        # setup_indices = []
        while (time_in_waveform > self.waveform_length).any():
            # setup_indices.append(np.where(self.time_in_waveform > self.waveform_length)[0][0])
            time_in_waveform[time_in_waveform > self.waveform_length] -= self.waveform_length
            self.setup_waveform()

        t = time_in_waveform / self.waveform_length

        lte_te = self.E0 * np.exp(self.alpha * t) * np.sin(self.omega * t)
        gt_te = (-np.exp(-self.epsilon * (t - self.Te)) + self.shift) / self.delta
        out = np.where(t > self.Te, gt_te, lte_te)

        voice_loudness = math.pow(self.tenseness, 0.25)
        out *= voice_loudness

        if randomize:
            noise = 1.0 * np.random.random(size=out.shape) - 0.5  # FIXME Test this...
        else:
            noise = 1.0 * 0.5 - 0.5

        aspiration = (1 - math.sqrt(self.tenseness)) * 0.2 * noise

        aspiration *= 0.2

        out += aspiration

        return out * self.intensity

    def compute(self, randomize: bool = True) -> float:

        self.values = self.compute_np(randomize)

        return self.values
