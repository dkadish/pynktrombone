import numpy as np
np.seterr(all='raise')

from tools import move_towards
from transient import MAX_TRANSIENTS, TransientPool, Transient

zeros = lambda n: np.zeros(n)

N = 44
NOSE_LENGTH = 28
NOSE_START = 17

class Tract:

    def __init__(self, samplerate):
        # FIXME unused
        # self.fade

        # Setup arrays
        self.diameter = zeros(N)
        self.rest_diameter = zeros(N)
        self.target_diameter = zeros(N)
        self.R = zeros(N)
        self.L = zeros(N)
        self.reflection = zeros(N+1)
        self.new_reflection = zeros(N+1)
        self.junction_outL = zeros(N+1)
        self.junction_outR = zeros(N+1)
        self.A = zeros(N)

        self.noseL = zeros(NOSE_LENGTH)
        self.noseR = zeros(NOSE_LENGTH)
        self.nose_junc_outL = zeros(NOSE_LENGTH+1)
        self.nose_junc_outR = zeros(NOSE_LENGTH+1)
        self.nose_reflection = zeros(NOSE_LENGTH+1)
        self.nose_diameter = zeros(NOSE_LENGTH)
        self.noseA = zeros(NOSE_LENGTH)

        # Default parameters
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
        for i in range(N):
            if i < (7.0 * N / 44.0 - 0.5):
                diameter = 0.6
            elif i < (12.0 * N / 44.0):
                diameter = 1.1
            else:
                diameter = 1.5

            self.diameter[i] = self.rest_diameter[i] = self.target_diameter[i] = diameter

        # Set up the nasal passage parameters
        for i in range(NOSE_LENGTH):
            d = 2.0 * (i / NOSE_LENGTH)
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

    def _calc_scattering_junctions(self, excitation, xfade_coeff):
        # Calculate Scattering Junctions
        self.junction_outR[0] = self.L[0] * self.glottal_reflection + excitation
        self.junction_outL[N] = self.R[N - 1] * self.lip_reflection

        r = self.reflection[1:N] * (1 - xfade_coeff) + self.new_reflection[1:N] * xfade_coeff
        w = r * (self.R[:-1] + self.L[1:])
        self.junction_outR[1:N] = self.R[:-1] - w
        self.junction_outL[1:N] = self.L[1:] + w

    def _calc_scattering_for_nose(self, xfade_coeff):
        # Calculate Scattering for Nose
        i = NOSE_START
        r = self.new_reflection_left * (1 - xfade_coeff) + self.reflection_left * xfade_coeff
        self.junction_outL[i] = r * self.R[i - 1] + (1 + r) * (self.noseL[0] + self.L[i])
        r = self.new_reflection_right * (1 - xfade_coeff) + self.reflection_right * xfade_coeff
        self.junction_outR[i] = r * self.L[i] + (1 + r) * (self.R[i - 1] + self.noseL[0])
        r = self.new_reflection_nose * (1 - xfade_coeff) + self.reflection_nose * xfade_coeff
        self.nose_junc_outR[0] = r * self.noseL[0] + (1 + r) * (self.L[i] + self.R[i - 1])

    def _update_delay_lines_and_lip(self):
        # Update Left/Right delay lines and set lip output
        self.R[:N] = self.junction_outR[:N] * 0.999
        self.L[:N] = self.junction_outL[1:N + 1] * 0.999

        self.lip_output = self.R[N - 1]

    def _calc_nose_scattering_jcts(self):
        # Calculate Nose Scattering Junctions
        self.nose_junc_outL[NOSE_LENGTH] = self.noseR[NOSE_LENGTH - 1] * self.lip_reflection

        w = self.nose_reflection[1:NOSE_LENGTH] * (self.noseR[:NOSE_LENGTH - 1] + self.noseL[1:NOSE_LENGTH])
        self.nose_junc_outR[1:NOSE_LENGTH] = self.noseR[:NOSE_LENGTH - 1] - w
        self.nose_junc_outL[1:NOSE_LENGTH] = self.noseL[1:NOSE_LENGTH] + w

    def _update_nose_delay_and_output(self):
        # Update Nose Left/Right delay lines and set nose output
        self.noseR[:NOSE_LENGTH] = self.nose_junc_outR[:NOSE_LENGTH]
        self.noseL[:NOSE_LENGTH] = self.nose_junc_outL[1:NOSE_LENGTH + 1]

        self.nose_output = self.noseR[NOSE_LENGTH - 1]

    def compute(self, excitation: float, xfade_coeff: float) -> None:  # in > excitation, lambda > xfade_coeff
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

        self._calc_scattering_junctions(excitation, xfade_coeff)
        self._calc_scattering_for_nose(xfade_coeff)
        self._update_delay_lines_and_lip()
        self._calc_nose_scattering_jcts()
        self._update_nose_delay_and_output()

    def calculate_nose_reflections(self):
        self.noseA[:NOSE_LENGTH] = self.nose_diameter[:NOSE_LENGTH] * self.nose_diameter[:NOSE_LENGTH]

        self.nose_reflection[1:NOSE_LENGTH] = (self.noseA[:NOSE_LENGTH-1] - self.noseA[1:NOSE_LENGTH]) / (self.noseA[:NOSE_LENGTH-1] + self.noseA[1:NOSE_LENGTH])

    def calculate_reflections(self):
        # self.A[i] = self.diameter[i] * self.diameter[i]
        self.A = np.power(self.diameter,2)

        self.reflection[1:N] = self.new_reflection[1:N]
        self.new_reflection[1:N][self.A[1:N] == 0] = 0.999
        non_zero = (self.A[:N-1] - self.A[1:N])/(self.A[:N-1] + self.A[1:N])
        self.new_reflection[1:N][self.A[1:N] != 0] = non_zero[self.A[1:N] != 0]

        self.reflection_left = self.new_reflection_left
        self.reflection_right = self.new_reflection_right
        self.reflection_nose = self.new_reflection_nose

        total = self.A[NOSE_START] + self.A[NOSE_START + 1] + self.noseA[0]
        self.new_reflection_left = (2.0 * self.A[NOSE_START] - total) / total
        self.new_reflection_right = (2.0 * self.A[NOSE_START + 1] - total) / total
        self.new_reflection_nose = (2.0 * self.noseA[0] - total) / total

    def reshape(self):
        current_obstruction = -1
        amount = self.block_time * self.movement_speed

        for i in range(N):
            diameter = self.diameter[i]
            target_diameter = self.target_diameter[i]

            if diameter < 0.001: current_obstruction = i

            if i < NOSE_START:
                slow_return = 0.6
            elif i >= self.tip_start:
                slow_return = 1.0
            else:
                slow_return = 0.6 + 0.4 * (i - NOSE_START) / (self.tip_start - NOSE_START)

            self.diameter[i] = move_towards(diameter, target_diameter, slow_return * amount, 2 * amount)

        if self.last_obstruction > -1 and current_obstruction == -1 and self.noseA[0] < 0.05:
            self.tpool.append(self.last_obstruction)

        self.last_obstruction = current_obstruction

        self.nose_diameter[0] = move_towards(self.nose_diameter[0], self.velum_target,
                                             amount * 0.25, amount * 0.1)
        self.noseA[0] = self.nose_diameter[0] * self.nose_diameter[0]


class TractNP:

    def __init__(self, samplerate):
        # FIXME unused
        # self.fade

        '''A note on variable names
        ALL_CAPS: Static throughout execution.
        _starts_with_underscore: updated once per sample generation (512 runs)
        normal: updated each cycle potentially.
        '''

        # Setup arrays
        self.diameter = zeros(N)
        self.rest_diameter = zeros(N)
        self.target_diameter = zeros(N)
        self.R = zeros(N)
        self.L = zeros(N)
        self._reflection = zeros(N + 1)
        self._new_reflection = zeros(N + 1)
        self.junction_outL = zeros(N+1)
        self.junction_outR = zeros(N+1)
        self.A = zeros(N)

        self.noseL = zeros(NOSE_LENGTH)
        self.noseR = zeros(NOSE_LENGTH)
        self.nose_junc_outL = zeros(NOSE_LENGTH+1)
        self.nose_junc_outR = zeros(NOSE_LENGTH+1)
        self.nose_reflection = zeros(NOSE_LENGTH+1)
        self.nose_diameter = zeros(NOSE_LENGTH)
        self.noseA = zeros(NOSE_LENGTH)

        # Default parameters
        self._reflection_left = 0.0
        self._reflection_right = 0.0
        self._reflection_nose = 0.0
        self._new_reflection_left = 0.0
        self._new_reflection_right = 0.0
        self._new_reflection_nose = 0.0
        self.velum_target = 0.01
        self.GLOTTAL_REFLECTION = 0.75
        self.LIP_REFLECTION = -0.85
        self.last_obstruction = -1
        self.movement_speed = 15
        self.lip_output = 0
        self.nose_output = 0
        self.tip_start = 32

        # FIXME Can be redone with array operations.
        # Set up the vocal tract diameters
        for i in range(N):
            if i < (7.0 * N / 44.0 - 0.5):
                diameter = 0.6
            elif i < (12.0 * N / 44.0):
                diameter = 1.1
            else:
                diameter = 1.5

            self.diameter[i] = self.rest_diameter[i] = self.target_diameter[i] = diameter

        # Set up the nasal passage parameters
        for i in range(NOSE_LENGTH):
            d = 2.0 * (i / NOSE_LENGTH)
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

    def _calc_scattering_junctions(self, excitation):
        # Calculate Scattering Junctions
        self.junction_outR[0] = self.L[0] * self.GLOTTAL_REFLECTION + excitation
        self.junction_outL[N] = self.R[N - 1] * self.LIP_REFLECTION

        w = r * (self.R[:-1] + self.L[1:])
        self.junction_outR[1:N] = self.R[:-1] - w
        self.junction_outL[1:N] = self.L[1:] + w

    def _calc_scattering_for_nose(self, r_outL, r_outR, r_nj):
        # Calculate Scattering for Nose
        j = NOSE_START
        self.junction_outL[j] = r_outL * self.R[j - 1] + (1 + r_outL) * (self.noseL[0] + self.L[j])
        self.junction_outR[j] = r_outR * self.L[j] + (1 + r_outR) * (self.R[j - 1] + self.noseL[0])
        self.nose_junc_outR[0] = r_nj * self.noseL[0] + (1 + r_nj) * (self.L[j] + self.R[j - 1])

    def _update_delay_lines_and_lip(self):
        # Update Left/Right delay lines and set lip output
        self.R[:N] = self.junction_outR[:N] * 0.999
        self.L[:N] = self.junction_outL[1:N + 1] * 0.999

        self.lip_output = self.R[N - 1]

    def _calc_nose_scattering_jcts(self):
        # Calculate Nose Scattering Junctions
        self.nose_junc_outL[NOSE_LENGTH] = self.noseR[NOSE_LENGTH - 1] * self.LIP_REFLECTION

        w = self.nose_reflection[1:NOSE_LENGTH] * (self.noseR[:NOSE_LENGTH-1] + self.noseL[1:NOSE_LENGTH])
        self.nose_junc_outR[1:NOSE_LENGTH] = self.noseR[:NOSE_LENGTH-1] - w
        self.nose_junc_outL[1:NOSE_LENGTH] = self.noseL[1:NOSE_LENGTH] + w

    def _update_nose_delay_and_output(self):

        # Update Nose Left/Right delay lines and set nose output
        self.noseR[:NOSE_LENGTH] = self.nose_junc_outR[:NOSE_LENGTH]
        self.noseL[:NOSE_LENGTH] = self.nose_junc_outL[1:NOSE_LENGTH+1]

        self.nose_output = self.noseR[NOSE_LENGTH - 1]

    def compute(self, excitation: np.array, xfade_coeff: float) -> None:  # in > excitation, lambda > xfade_coeff
        r = self._reflection[1:N] * (1 - xfade_coeff) + self._new_reflection[1:N] * xfade_coeff
        r_outL = self._new_reflection_left * (1 - xfade_coeff) + self._reflection_left * xfade_coeff
        r_outR = self._new_reflection_right * (1 - xfade_coeff) + self._reflection_right * xfade_coeff
        r_nj = self._new_reflection_nose * (1 - xfade_coeff) + self._reflection_nose * xfade_coeff

        # Process Transients
        pool = self.tpool
        current_size = pool.size
        n = pool.root
        for k in range(current_size):
            amp = n.strength * pow(2.0, -1.0 * n.exponent * n.time_alive)
            self.L[n.position] += amp * 0.5
            self.R[n.position] += amp * 0.5
            n.time_alive += self.T * 0.5
            if n.time_alive > n.lifetime:
                pool.remove(n.id)
            n = n.next

        self._calc_scattering_junctions(excitation)
        self._calc_scattering_for_nose(r_outL, r_outR, r_nj)
        self._update_delay_lines_and_lip()
        self._calc_nose_scattering_jcts()
        self._update_nose_delay_and_output()

    def calculate_nose_reflections(self):
        self.noseA[:NOSE_LENGTH] = self.nose_diameter[:NOSE_LENGTH] * self.nose_diameter[:NOSE_LENGTH]

        self.nose_reflection[1:NOSE_LENGTH] = (self.noseA[:NOSE_LENGTH-1] - self.noseA[1:NOSE_LENGTH]) / (self.noseA[:NOSE_LENGTH-1] + self.noseA[1:NOSE_LENGTH])

    def calculate_reflections(self):
        # self.A[i] = self.diameter[i] * self.diameter[i]
        self.A = np.power(self.diameter,2)

        self._reflection[1:N] = self._new_reflection[1:N]
        self._new_reflection[1:N][self.A[1:N] == 0] = 0.999
        non_zero = (self.A[:N-1] - self.A[1:N])/(self.A[:N-1] + self.A[1:N])
        self._new_reflection[1:N][self.A[1:N] != 0] = non_zero[self.A[1:N] != 0]

        self._reflection_left = self._new_reflection_left
        self._reflection_right = self._new_reflection_right
        self._reflection_nose = self._new_reflection_nose

        total = self.A[NOSE_START] + self.A[NOSE_START + 1] + self.noseA[0]
        self._new_reflection_left = (2.0 * self.A[NOSE_START] - total) / total
        self._new_reflection_right = (2.0 * self.A[NOSE_START + 1] - total) / total
        self._new_reflection_nose = (2.0 * self.noseA[0] - total) / total

    def reshape(self):
        current_obstruction = -1
        amount = self.block_time * self.movement_speed

        for i in range(N):
            diameter = self.diameter[i]
            target_diameter = self.target_diameter[i]

            if diameter < 0.001: current_obstruction = i

            if i < NOSE_START:
                slow_return = 0.6
            elif i >= self.tip_start:
                slow_return = 1.0
            else:
                slow_return = 0.6 + 0.4 * (i - NOSE_START) / (self.tip_start - NOSE_START)

            self.diameter[i] = move_towards(diameter, target_diameter, slow_return * amount, 2 * amount)

        if self.last_obstruction > -1 and current_obstruction == -1 and self.noseA[0] < 0.05:
            self.tpool.append(self.last_obstruction)

        self.last_obstruction = current_obstruction

        self.nose_diameter[0] = move_towards(self.nose_diameter[0], self.velum_target,
                                             amount * 0.25, amount * 0.1)
        self.noseA[0] = self.nose_diameter[0] * self.nose_diameter[0]
