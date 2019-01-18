import math
from random import random

from tools import move_towards, sp_data


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

        self.setup_waveform(0.0)

        self.T = 1.0 / sr

    def setup_waveform(self, lmbda: float):

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

        self.Rd = 3 * (1 - self.tenseness)
        self.waveform_length = 1.0 / self.freq

        Rd = self.Rd
        if Rd < 0.5:
            Rd = 0.5
        elif Rd > 2.7:
            Rd = 2.7

        Ra = -0.01 + 0.048 * Rd
        Rk = 0.224 + 0.118 * Rd
        Rg = (Rk / 4) * (0.5 + 1.2 * Rk) / (0.11 * Rd - Ra * (0.5 + 1.2 * Rk))

        Ta = Ra
        Tp = 1.0 / (2.0 * Rg)
        Te = Tp + Tp * Rk

        epsilon = 1.0 / Ta
        shift = math.exp(-epsilon * (1.0 - Te))
        delta = 1 - shift

        rhs_integral = (1.0 / epsilon) * (shift - 1.0) + (1.0 - Te) * shift
        rhs_integral = rhs_integral / delta
        lower_integral = -(Te - Tp) / 2 + rhs_integral
        upper_integral = -lower_integral

        omega = math.pi / Tp
        s = math.sin(omega * Te)

        y = -math.pi * s * upper_integral / (Tp * 2)
        z = math.log(y)
        alpha = z / (Tp / 2 - Te)
        E0 = -1 / (s * math.exp(alpha * Te))

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
                                      block_time / self.release_time);

    def compute(self, lmbda: float, randomize: bool = True) -> float:
        out = 0.0;
        self.time_in_waveform += self.T

        if self.time_in_waveform > self.waveform_length:
            self.time_in_waveform -= self.waveform_length
            self.setup_waveform(lmbda)

        t = (self.time_in_waveform / self.waveform_length)

        if t > self.Te:
            out = (-math.exp(-self.epsilon * (t - self.Te)) + self.shift) / self.delta
        else:
            out = self.E0 * math.exp(self.alpha * t) * math.sin(self.omega * t)

        voice_loudness = pow(self.tenseness, 0.25)
        out *= voice_loudness

        if randomize:
            noise = 1.0 * random() - 0.5 #FIXME Test this...
        else:
            noise = 1.0 * 0.5 - 0.5
        # ################################################################################################################
        # # Corresponds to https://github.com/jamesstaub/pink-trombone-osc/blob/d700292127f31b73b44103c0e8dc4865a3cca651/src/audio/glottis.js#L192
        # noise = 1.0 * ((SPFLOAT) sp_rand(sp) / SP_RANDMAX) - 0.5
        #
        # # this.getNoiseModulator() * noiseSource
        # voiced = 0.1 + 0.2 * max([0.0, math.sin(math.pi * 2.0 * self.time_in_waveform / self.waveform_length)])
        # noiseModulator = self.tenseness * self.intensity * voiced + (1 - self.tenseness * self.intensity) * 0.3
        # ################################################################################################################

        #JS: var aspiration = this.intensity * (1 - Math.sqrt( this.UITenseness)) * this.getNoiseModulator() * noiseSource;
        aspiration = (1 - math.sqrt(self.tenseness)) *0.2 * noise

        aspiration *= 0.2

        out += aspiration

        return out * self.intensity