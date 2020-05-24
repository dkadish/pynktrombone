from math import exp, sin, log, sqrt
from random import random

from . import voc


class Glottis:

    def __init__(self, sr: float):
        self.freq: float = 140  # 140Hz frequency by default
        self.tenseness: float = 0.6  # value between 0 and 1
        self.Rd: float
        self.waveform_length: float
        self.time_in_waveform: float = 0
        self.alpha: float
        self.E0: float
        self.epsilon: float
        self.shift: float
        self.delta: float
        self.Te: float
        self.omega: float
        self.T: float = 1.0 / sr  # big T

        self.setup_waveform(0)

    # static void glottis_init(Glottis *self, SPFLOAT samplerate)
    # CHANGE: self is not a pointer, is returned from fn
    # def glottis_init(self: Glottis, samplerate: float) -> Glottis:

    # static void glottis_setup_waveform(Glottis *self, SPFLOAT lmbd)
    # CHANGE: self is not a pointer, is returned from fn
    def setup_waveform(self, lmbd: float):
        Rd: float
        Ra: float
        Rk: float
        Rg: float

        Ta: float
        Tp: float
        Te: float

        epsilon: float
        shift: float
        delta: float
        rhs_integral: float

        lower_integral: float
        upper_integral: float

        omega: float
        s: float
        y: float
        z: float

        alpha: float
        E0: float

        self.Rd = 3 * (1 - self.tenseness)
        self.waveform_length = 1.0 / self.freq

        Rd = self.Rd
        if (Rd < 0.5): Rd = 0.5
        if (Rd > 2.7): Rd = 2.7

        Ra = -0.01 + 0.048 * Rd
        Rk = 0.224 + 0.118 * Rd
        Rg = (Rk / 4) * (0.5 + 1.2 * Rk) / (0.11 * Rd - Ra * (0.5 + 1.2 * Rk))

        Ta = Ra
        Tp = float(1.0 / (2 * Rg))
        Te = Tp + Tp * Rk

        epsilon = float(1.0 / Ta)
        shift = exp(-epsilon * (1 - Te))
        delta = 1 - shift

        rhs_integral = float((1.0 / epsilon) * (shift - 1) + (1 - Te) * shift)
        rhs_integral = rhs_integral / delta
        lower_integral = - (Te - Tp) / 2 + rhs_integral
        upper_integral = -lower_integral

        omega = voc.M_PI / Tp
        s = sin(omega * Te)

        y = -voc.M_PI * s * upper_integral / (Tp * 2)
        z = log(y)
        alpha = z / (Tp / 2 - Te)
        E0 = -1 / (s * exp(alpha * Te))

        self.alpha = alpha
        self.E0 = E0
        self.epsilon = epsilon
        self.shift = shift
        self.delta = delta
        self.Te = Te
        self.omega = omega

    # static SPFLOAT glottis_compute(sp_data *sp, Glottis *self, SPFLOAT lmbd)
    # CHANGE: sp is not a pointer, is returned from fn
    # CHANGE: self is not a pointer, is returned from fn
    def compute(self, lmbd: float) -> float:
        intensity: float = 1.0

        self.time_in_waveform += self.T

        if self.time_in_waveform > self.waveform_length:
            self.time_in_waveform -= self.waveform_length
            self.setup_waveform(lmbd)

        t = (self.time_in_waveform / self.waveform_length)

        if t > self.Te:
            out = (-exp(-self.epsilon * (t - self.Te)) + self.shift) / self.delta
        else:
            out = self.E0 * exp(self.alpha * t) * sin(self.omega * t)

        noise = 2.0 * random() - 1

        aspiration = intensity * (1 - sqrt(self.tenseness)) * 0.3 * noise

        aspiration *= 0.2

        out += aspiration

        return out
