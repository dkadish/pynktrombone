import numpy as np
np.seterr(all='raise')

from tools import move_towards
from transient import MAX_TRANSIENTS, TransientPool, Transient

zeros = lambda n: np.zeros(n)


class Tract:

    def __init__(self, samplerate):
        # FIXME unused
        # self.fade

        # Setup arrays
        self.diameter = zeros(44)
        self.rest_diameter = zeros(44)
        self.target_diameter = zeros(44)
        self.R = zeros(44)
        self.L = zeros(44)
        self.reflection = zeros(45)
        self.new_reflection = zeros(45)
        self.junction_outL = zeros(45)
        self.junction_outR = zeros(45)
        self.A = zeros(44)

        self.noseL = zeros(28)
        self.noseR = zeros(28)
        self.nose_junc_outL = zeros(29)
        self.nose_junc_outR = zeros(29)
        self.nose_reflection = zeros(29)
        self.nose_diameter = zeros(28)
        self.noseA = zeros(28)

        # Default parameters
        self.n = 44
        self.nose_length = 28
        self.nose_start = 17

        self.reflection_left = 0.0
        self.reflection_right = 0.0
        self.reflection_nose = 0.0
        self.new_reflection_left = 0.0
        self.new_reflection_right = 0.0
        self.new_reflection_nose = 0.0
        self.velum_target = 0.01
        self.glottal_reflection = 0.75
        self.lip_reflection = -0.85
        self.last_obstruction = -1
        self.movement_speed = 15
        self.lip_output = 0
        self.nose_output = 0
        self.tip_start = 32

        # FIXME Can be redone with array operations.
        # Set up the vocal tract diameters
        for i in range(self.n):
            if i < (7.0 * self.n / 44.0 - 0.5):
                diameter = 0.6
            elif i < (12.0 * self.n / 44.0):
                diameter = 1.1
            else:
                diameter = 1.5

            self.diameter[i] = self.rest_diameter[i] = self.target_diameter[i] = diameter

        # Set up the nasal passage parameters
        for i in range(self.nose_length):
            d = 2.0 * (i / self.nose_length)
            if d < 1:
                diameter = 0.4 + 1.6 * d
            else:
                diameter = 0.5 + 1.5 * (2 - d)

            diameter = min(diameter, 1.9)
            self.nose_diameter[i] = diameter

        self.calculate_reflections()
        self.calculate_nose_reflections()

        self.nose_diameter[0] = self.velum_target

        self.block_time = 512.0 / samplerate
        self.T = 1.0 / samplerate

        self.tpool = TransientPool()
        self.tpool.size = 0
        self.tpool.next_free = 0
        for i in range(MAX_TRANSIENTS):
            tr = Transient(position=0, id=i, next=None)
            self.tpool.pool.append(tr)
            self.tpool.pool[i].is_free = 1
            self.tpool.pool[i].time_alive = 0
            self.tpool.pool[i].strength = 0
            self.tpool.pool[i].exponent = 0

    def compute(self, excitation, xfade_coeff):  # in > excitation, lambda > xfade_coeff
        # Process Transients
        pool = self.tpool
        current_size = pool.size
        n = pool.root
        for i in range(current_size):
            amp = n.strength * pow(2.0, -1.0 * n.exponent * n.time_alive)
            self.L[n.position] += amp * 0.5
            self.R[n.position] += amp * 0.5
            n.time_alive += self.T * 0.5
            if n.time_alive > n.lifetime:
                pool.remove(n.id)
            n = n.next

        # Calculate Scattering Junctions
        self.junction_outR[0] = self.L[0] * self.glottal_reflection + excitation
        self.junction_outL[self.n] = self.R[self.n - 1] * self.lip_reflection

        r = self.reflection[1:self.n] * (1-xfade_coeff) + self.new_reflection[1:self.n] * xfade_coeff
        w = r * (self.R[:-1] + self.L[1:])
        self.junction_outR[1:self.n] = self.R[:-1] - w
        self.junction_outL[1:self.n] = self.L[1:] + w

        # Calculate Scattering for Nose
        i = self.nose_start
        r = self.new_reflection_left * (1 - xfade_coeff) + self.reflection_left * xfade_coeff
        self.junction_outL[i] = r * self.R[i - 1] + (1 + r) * (self.noseL[0] + self.L[i])
        r = self.new_reflection_right * (1 - xfade_coeff) + self.reflection_right * xfade_coeff
        self.junction_outR[i] = r * self.L[i] + (1 + r) * (self.R[i - 1] + self.noseL[0])
        r = self.new_reflection_nose * (1 - xfade_coeff) + self.reflection_nose * xfade_coeff
        self.nose_junc_outR[0] = r * self.noseL[0] + (1 + r) * (self.L[i] + self.R[i - 1])

        # Update Left/Right delay lines and set lip output
        self.R[:self.n] = self.junction_outR[:self.n] * 0.999
        self.L[:self.n] = self.junction_outL[1:self.n+1] * 0.999

        self.lip_output = self.R[self.n - 1]

        # Calculate Nose Scattering Junctions
        self.nose_junc_outL[self.nose_length] = self.noseR[self.nose_length - 1] * self.lip_reflection

        w = self.nose_reflection[1:self.nose_length] * (self.noseR[:self.nose_length-1] + self.noseL[1:self.nose_length])
        self.nose_junc_outR[1:self.nose_length] = self.noseR[:self.nose_length-1] - w
        self.nose_junc_outL[1:self.nose_length] = self.noseL[1:self.nose_length] + w

        # Update Nose Left/Right delay lines and set nose output
        self.noseR[:self.nose_length] = self.nose_junc_outR[:self.nose_length]
        self.noseL[:self.nose_length] = self.nose_junc_outL[1:self.nose_length+1]

        self.nose_output = self.noseR[self.nose_length - 1]

    def calculate_nose_reflections(self):
        for i in range(self.nose_length):
            self.noseA[i] = self.nose_diameter[i] * self.nose_diameter[i]

        for i in range(1, self.nose_length):
            self.nose_reflection[i] = (self.noseA[i - 1] - self.noseA[i]) / (self.noseA[i - 1] + self.noseA[i])

    def calculate_reflections(self):
        for i in range(self.n):
            self.A[i] = self.diameter[i] * self.diameter[i]

        for i in range(1, self.n):
            self.reflection[i] = self.new_reflection[i]
            if self.A[i] == 0:
                self.new_reflection[i] = 0.999
            else:
                self.new_reflection[i] = (self.A[i - 1] - self.A[i]) / (self.A[i - 1] + self.A[i])

        self.reflection_left = self.new_reflection_left
        self.reflection_right = self.new_reflection_right
        self.reflection_nose = self.new_reflection_nose

        total = self.A[self.nose_start] + self.A[self.nose_start + 1] + self.noseA[0]
        self.new_reflection_left = (2.0 * self.A[self.nose_start] - total) / total
        self.new_reflection_right = (2.0 * self.A[self.nose_start + 1] - total) / total
        self.new_reflection_nose = (2.0 * self.noseA[0] - total) / total

    def reshape(self):
        current_obstruction = -1
        amount = self.block_time * self.movement_speed

        for i in range(self.n):
            diameter = self.diameter[i]
            target_diameter = self.target_diameter[i]

            if diameter < 0.001: current_obstruction = i

            if i < self.nose_start:
                slow_return = 0.6
            elif i >= self.tip_start:
                slow_return = 1.0
            else:
                slow_return = 0.6 + 0.4 * (i - self.nose_start) / (self.tip_start - self.nose_start)

            self.diameter[i] = move_towards(diameter, target_diameter, slow_return * amount, 2 * amount)

        if self.last_obstruction > -1 and current_obstruction == -1 and self.noseA[0] < 0.05:
            self.tpool.append(self.last_obstruction)

        self.last_obstruction = current_obstruction

        self.nose_diameter[0] = move_towards(self.nose_diameter[0], self.velum_target,
                                             amount * 0.25, amount * 0.1)
        self.noseA[0] = self.nose_diameter[0] * self.nose_diameter[0]
