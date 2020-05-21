# include <stdlib.h>
# include <math.h>
# include <string.h>
# include "soundpipe.h"
from math import exp, sin, log, sqrt, cos

from typing import List, Tuple

M_PI = 3.14159265358979323846

# include "voc.h"

# ifndef MIN
# define MIN(A,B) ((A) < (B) ? (A) : (B))
# endif

# ifndef MAX
# define MAX(A,B) ((A) > (B) ? (A) : (B))
# endif

EPSILON = 1.0e-38

MAX_TRANSIENTS = 4


#################################################################
# TODO Replace Dummies
# DUMMY CLASS
class sp_data:
    pass


# DUMMY METHOD
def sp_rand(sp):
    pass


# DUMMY VAR
SP_RANDMAX = 0


#################################################################

class glottis:

    def __init__(self):
        self.freq: float

        self.freq: float
        self.tenseness: float
        self.Rd: float
        self.waveform_length: float
        self.time_in_waveform: float
        self.alpha: float
        self.E0: float
        self.epsilon: float
        self.shift: float
        self.delta: float
        self.Te: float
        self.omega: float
        self.T: float


class transient:
    def __init__(self):
        self.position: int
        self.time_alive: float
        self.lifetime: float
        self.strength: float
        self.exponent: float
        self.is_free: str
        self.id: int
        self.next: transient  # PTR


class transient_pool:
    def __init__(self):
        self.pool: List[transient]  # Should be limited to MAX_TRANSIENTS
        self.root: transient  # PTR
        self.size: int
        self.next_free: int


class tract:
    def __init__(self):
        self.n: int

        self.diameter: List[float]  # len = 44
        self.rest_diameter: List[float]  # len = 44
        self.target_diameter: List[float]  # len = 44
        self.new_diameter: List[float]  # len = 44
        self.R: List[float]  # len = 44
        self.L: List[float]  # len = 44
        self.reflection: List[float]  # len = 44
        self.new_reflection: List[float]  # len = 44
        self.junction_outL: List[float]  # len = 44
        self.junction_outR: List[float]  # len = 44
        self.A: List[float]  # len = 44

        self.nose_length: int

        self.nose_start: int

        self.tip_start: int
        self.noseL: List[float]  # len = 28
        self.noseR: List[float]  # len = 28
        self.nose_junc_outL: List[float]  # len = 29
        self.nose_junc_outR: List[float]  # len = 29
        self.nose_reflection: List[float]  # len = 29
        self.nose_diameter: List[float]  # len = 28
        self.noseA: List[float]  # len = 28

        self.reflection_left: float
        self.reflection_right: float
        self.reflection_nose: float

        self.new_reflection_left: float
        self.new_reflection_right: float
        self.new_reflection_nose: float

        self.velum_target: float

        self.glottal_reflection: float
        self.lip_reflection: float
        self.last_obstruction: int
        self.fade: float
        self.movement_speed: float
        self.lip_output: float
        self.nose_output: float
        self.block_time: float

        self.tpool: transient_pool
        self.T: float


class sp_voc:
    def __init__(self):
        self.glot: glottis  # The Glottis
        self.tr: tract  # The Vocal Tract
        self.buf: List[float]  # len = 512
        self._counter: int

    @property
    def frequency(self) -> float:
        return self.glot.freq

    @frequency.setter
    def frequency(self, f: float):
        self.glot.freq = f

    # void sp_voc_set_frequency(sp_voc *voc, SPFLOAT freq)
    # {
    #     voc->glot.freq = freq;
    # }
    #
    #
    # SPFLOAT * sp_voc_get_frequency_ptr(sp_voc *voc)
    # {
    #     return &voc->glot.freq;
    # }

    @property
    def tract_diameters(self) -> float:
        return self.tr.target_diameter

    # SPFLOAT* sp_voc_get_tract_diameters(sp_voc *voc)
    # {
    #     return voc->tr.target_diameter;
    # }

    @property
    def current_tract_diameters(self) -> float:
        return self.tr.diameter

    # SPFLOAT* sp_voc_get_current_tract_diameters(sp_voc *voc)
    # {
    #     return voc->tr.diameter;
    # }

    @property
    def tract_size(self) -> int:
        return self.tr.n

    # int sp_voc_get_tract_size(sp_voc *voc)
    # {
    #     return voc->tr.n;
    # }

    @property
    def nose_diameters(self) -> float:
        return self.tr.nose_diameter

    # SPFLOAT* sp_voc_get_nose_diameters(sp_voc *voc)
    # {
    #     return voc->tr.nose_diameter;
    # }

    @property
    def nose_size(self) -> int:
        return self.tr.nose_length

    # int sp_voc_get_nose_size(sp_voc *voc)
    # {
    #     return voc->tr.nose_length;
    # }

    @property
    def counter(self) -> int:
        return self._counter

    # int sp_voc_get_counter(sp_voc *voc)
    # {
    #     return voc->counter;
    # }

    @property
    def tenseness(self) -> float:
        return self.glot.tenseness

    @tenseness.setter
    def tenseness(self, t: float):
        self.glot.tenseness = t

    # SPFLOAT * sp_voc_get_tenseness_ptr(sp_voc *voc)
    # {
    #     return &voc->glot.tenseness;
    # }

    # void sp_voc_set_tenseness(sp_voc *voc, SPFLOAT tenseness)
    # {
    #     voc->glot.tenseness = tenseness;
    # }

    @property
    def velum(self) -> float:
        return self.tr.velum_target

    @tenseness.setter
    def velum(self, t: float):
        self.tr.velum_target = t

    # void sp_voc_set_velum(sp_voc *voc, SPFLOAT velum)
    # {
    #     voc->tr.velum_target = velum;
    # }
    #
    #
    # SPFLOAT *sp_voc_get_velum_ptr(sp_voc *voc)
    # {
    #     return &voc->tr.velum_target;
    # }


# static void glottis_setup_waveform(glottis *glot, SPFLOAT lmbd)
# CHANGE: glot is not a pointer, is returned from fn
def glottis_setup_waveform(glot: glottis, lmbd: float) -> glottis:
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

    glot.Rd = 3 * (1 - glot.tenseness)
    glot.waveform_length = 1.0 / glot.freq

    Rd = glot.Rd
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

    omega = M_PI / Tp
    s = sin(omega * Te)

    y = -M_PI * s * upper_integral / (Tp * 2)
    z = log(y)
    alpha = z / (Tp / 2 - Te)
    E0 = -1 / (s * exp(alpha * Te))

    glot.alpha = alpha
    glot.E0 = E0
    glot.epsilon = epsilon
    glot.shift = shift
    glot.delta = delta
    glot.Te = Te
    glot.omega = omega

    return glot


# static void glottis_init(glottis *glot, SPFLOAT sr)
# CHANGE: glot is not a pointer, is returned from fn
def glottis_init(glot: glottis, sr: float) -> glottis:
    glot.freq = 140  # 140Hz frequency by default
    glot.tenseness = 0.6  # value between 0 and 1
    glot.T = 1.0 / sr  # big T
    glot.time_in_waveform = 0
    return glottis_setup_waveform(glot, 0)


# static SPFLOAT glottis_compute(sp_data *sp, glottis *glot, SPFLOAT lmbd)
# CHANGE: sp is not a pointer, is returned from fn
# CHANGE: glot is not a pointer, is returned from fn
def glottis_compute(sp: sp_data, glot: glottis, lmbd: float) -> Tuple[sp_data, glottis, float]:
    out: float = 0.0
    aspiration: float
    noise: float
    t: float
    intensity: float = 1.0

    glot.time_in_waveform += glot.T

    if (glot.time_in_waveform > glot.waveform_length):
        glot.time_in_waveform -= glot.waveform_length
        glot = glottis_setup_waveform(glot, lmbd)

    t = (glot.time_in_waveform / glot.waveform_length)

    if (t > glot.Te):
        out = (-exp(-glot.epsilon * (t - glot.Te)) + glot.shift) / glot.delta
    else:
        out = glot.E0 * exp(glot.alpha * t) * sin(glot.omega * t)

    noise = 2.0 * float(sp_rand(sp) / SP_RANDMAX) - 1

    aspiration = intensity * (1 - sqrt(glot.tenseness)) * 0.3 * noise

    aspiration *= 0.2

    out += aspiration

    return sp, glot, out


# static void tract_calculate_reflections(tract *tr)
# CHANGE: tr is not a pointer, is returned from fn
def tract_calculate_reflections(tr: tract) -> tract:
    # TODO refactor rename i
    i: int
    _sum: float

    for i in range(tr.n):
        tr.A[i] = tr.diameter[i] * tr.diameter[i]
        # /* Calculate area from diameter squared*/

    for i in range(1, tr.n):
        tr.reflection[i] = tr.new_reflection[i]
        if tr.A[i] == 0:
            tr.new_reflection[i] = 0.999  # /* to prevent bad behavior if 0 */
        else:
            tr.new_reflection[i] = (tr.A[i - 1] - tr.A[i]) / (tr.A[i - 1] + tr.A[i])

    tr.reflection_left = tr.new_reflection_left
    tr.reflection_right = tr.new_reflection_right
    tr.reflection_nose = tr.new_reflection_nose

    _sum = tr.A[tr.nose_start] + tr.A[tr.nose_start + 1] + tr.noseA[0]
    tr.new_reflection_left = float(2 * tr.A[tr.nose_start] - _sum) / _sum
    tr.new_reflection_right = float(2 * tr.A[tr.nose_start + 1] - _sum) / _sum
    tr.new_reflection_nose = float(2 * tr.noseA[0] - _sum) / _sum


# static void tract_calculate_nose_reflections(tract *tr)
# CHANGE: tr is not a pointer, is returned from fn
def tract_calculate_nose_reflections(tr: tract) -> tract:
    # TODO refactor rename i
    i: int

    for i in range(tr.nose_length):
        tr.noseA[i] = tr.nose_diameter[i] * tr.nose_diameter[i]

    for i in range(1, tr.nose_length):
        tr.nose_reflection[i] = (tr.noseA[i - 1] - tr.noseA[i]) / (tr.noseA[i - 1] + tr.noseA[i])


# static int append_transient(transient_pool *pool, int position)
# CHANGE: pool is not a pointer, is returned from fn
def append_transient(pool: transient_pool, position: int) -> transient_pool:
    i: int
    free_id: int
    # transient *t
    t: transient

    free_id = pool.next_free
    if pool.size == MAX_TRANSIENTS: return pool

    if free_id == -1:
        for i in range(MAX_TRANSIENTS):
            if pool.pool[i].is_free:
                free_id = i
                break

    if free_id == -1: return pool

    t = transient()
    pool.pool[free_id] = t
    # t = &pool.pool[free_id]
    t.next = pool.root
    pool.root = t
    pool.size += 1
    t.is_free = 0
    t.time_alive = 0
    t.lifetime = 0.2
    t.strength = 0.3
    t.exponent = 200
    t.position = position
    pool.next_free = -1
    return pool


# static void remove_transient(transient_pool *pool, unsigned int id)
# CHANGE: pool is not a pointer, is returned from fn
def remove_transient(pool: transient_pool, id: int) -> transient_pool:
    i: int
    n: transient

    pool.next_free = id
    n = pool.root
    if id == n.id:
        pool.root = n.next
        pool.size -= 1
        return pool

    for i in range(pool.size):
        if n.next.id == id:
            pool.size -= 1
            n.next.is_free = 1
            n.next = n.next.next
            break
        n = n.next

    return pool


# static SPFLOAT move_towards(SPFLOAT current, SPFLOAT target,
#         SPFLOAT amt_up, SPFLOAT amt_down)
def move_towards(current: float, target: float, amt_up: float, amt_down: float) -> float:
    tmp: float
    if current < target:
        tmp = current + amt_up
        return min(tmp, target)

    else:
        tmp = current - amt_down
        return max(tmp, target)

    return 0.0


# static void tract_reshape(tract *tr)
# CHANGE: tr is not a pointer, is returned from fn
def tract_reshape(tr: tract) -> tract:
    amount: float
    slow_return: float
    diameter: float
    target_diameter: float
    i: int
    current_obstruction: int = -1
    amount = tr.block_time * tr.movement_speed

    for i in range(tr.n):
        slow_return = 0
        diameter = tr.diameter[i]
        target_diameter = tr.target_diameter[i]

        if diameter < 0.001: current_obstruction = i

        if i < tr.nose_start:
            slow_return = 0.6
        elif i >= tr.tip_start:
            slow_return = 1.0
        else:
            slow_return = 0.6 + 0.4 * (i - tr.nose_start) / (tr.tip_start - tr.nose_start)

        tr.diameter[i] = move_towards(diameter, target_diameter,
                                      slow_return * amount, 2 * amount)

    if tr.last_obstruction > -1 and current_obstruction == -1 and tr.noseA[0] < 0.05:
        tr.tpool = append_transient(tr.tpool, tr.last_obstruction)
    tr.last_obstruction = current_obstruction

    tr.nose_diameter[0] = move_towards(tr.nose_diameter[0], tr.velum_target,
                                       amount * 0.25, amount * 0.1)
    tr.noseA[0] = tr.nose_diameter[0] * tr.nose_diameter[0]

    return tract


def zeros(n):
    return [0.0 for _ in range(n)]


# static void tract_init(sp_data *sp, tract *tr)
# CHANGE: sp is not a pointer, is returned from fn
# CHANGE: tr is not a pointer, is returned from fn
def tract_init(sp: sp_data, tr: tract) -> Tuple[sp_data, tract]:
    i: int
    diameter: float
    d: float  # /* needed to set up diameter arrays */

    tr.n = 44;
    tr.nose_length = 28;
    tr.nose_start = 17;

    tr.reflection_left = 0.0;
    tr.reflection_right = 0.0;
    tr.reflection_nose = 0.0;
    tr.new_reflection_left = 0.0;
    tr.new_reflection_right = 0.0;
    tr.new_reflection_nose = 0.0;
    tr.velum_target = 0.01;
    tr.glottal_reflection = 0.75;
    tr.lip_reflection = -0.85;
    tr.last_obstruction = -1;
    tr.movement_speed = 15;
    tr.lip_output = 0;
    tr.nose_output = 0;
    tr.tip_start = 32;

    tr.diameter = zeros(tr.n)
    tr.rest_diameter = zeros(tr.n)
    tr.target_diameter = zeros(tr.n)
    tr.new_diameter = zeros(tr.n)
    tr.L = zeros(tr.n)
    tr.R = zeros(tr.n)
    tr.reflection = zeros((tr.n + 1))
    tr.new_reflection = zeros((tr.n + 1))
    tr.junction_outL = zeros((tr.n + 1))
    tr.junction_outR = zeros((tr.n + 1))
    tr.A = zeros(tr.n)
    tr.noseL = zeros(tr.nose_length)
    tr.noseR = zeros(tr.nose_length)
    tr.nose_junc_outL = zeros((tr.nose_length + 1))
    tr.nose_junc_outR = zeros((tr.nose_length + 1))
    tr.nose_diameter = zeros(tr.nose_length)
    tr.noseA = zeros(tr.nose_length)

    for i in range(tr.n):
        diameter = 0;
        if i < 7 * float(tr.n) / 44 - 0.5:
            diameter = 0.6;
        elif i < 12 * float(tr.n) / 44:
            diameter = 1.1;
        else:
            diameter = 1.5;

        tr.diameter[i] = tr.rest_diameter[i] = tr.target_diameter[i] = tr.new_diameter[i] = diameter;

    for i in range(tr.nose_length):
        d = 2 * (float(i) / tr.nose_length);
        if d < 1:
            diameter = 0.4 + 1.6 * d;
        else:
            diameter = 0.5 + 1.5 * (2 - d);

        diameter = min(diameter, 1.9);
        tr.nose_diameter[i] = diameter;

    tract_calculate_reflections(tr);
    tract_calculate_nose_reflections(tr);
    tr.nose_diameter[0] = tr.velum_target;

    tr.block_time = 512.0 / float(sp.sr)
    tr.T = 1.0 / float(sp.sr)

    tr.tpool.size = 0;
    tr.tpool.next_free = 0;
    for i in range(MAX_TRANSIENTS):
        tr.tpool.pool[i].is_free = 1;
        tr.tpool.pool[i].id = i;
        tr.tpool.pool[i].position = 0;
        tr.tpool.pool[i].time_alive = 0;
        tr.tpool.pool[i].strength = 0;
        tr.tpool.pool[i].exponent = 0;

    return sp, tract


# static void tract_compute(sp_data *sp, tract *tr,
#     SPFLOAT  in,
#     SPFLOAT  lmbd)
def tract_compute(sp: sp_data, tr: tract, _in: float, lmbd: float) -> Tuple[sp_data, tract]:
    r: float
    w: float
    i: int
    amp: float
    current_size: int
    # transient_pool *pool
    # transient *n
    pool: transient_pool
    n: transient

    pool = tr.tpool  # Python treats this as a reference, so this should be fine.
    current_size = pool.size
    n = pool.root
    for i in range(current_size):
        amp = n.strength * pow(2, -1.0 * n.exponent * n.time_alive)
        tr.L[n.position] += amp * 0.5
        tr.R[n.position] += amp * 0.5
        n.time_alive += tr.T * 0.5
        if n.time_alive > n.lifetime:
            remove_transient(pool, n.id)
        n = n.next

    tr.junction_outR[0] = tr.L[0] * tr.glottal_reflection + _in
    tr.junction_outL[tr.n] = tr.R[tr.n - 1] * tr.lip_reflection

    for i in range(1, tr.n):
        r = tr.reflection[i] * (1 - lmbd) + tr.new_reflection[i] * lmbd
        w = r * (tr.R[i - 1] + tr.L[i])
        tr.junction_outR[i] = tr.R[i - 1] - w
        tr.junction_outL[i] = tr.L[i] + w

    i = tr.nose_start
    r = tr.new_reflection_left * (1 - lmbd) + tr.reflection_left * lmbd
    tr.junction_outL[i] = r * tr.R[i - 1] + (1 + r) * (tr.noseL[0] + tr.L[i])
    r = tr.new_reflection_right * (1 - lmbd) + tr.reflection_right * lmbd
    tr.junction_outR[i] = r * tr.L[i] + (1 + r) * (tr.R[i - 1] + tr.noseL[0])
    r = tr.new_reflection_nose * (1 - lmbd) + tr.reflection_nose * lmbd
    tr.nose_junc_outR[0] = r * tr.noseL[0] + (1 + r) * (tr.L[i] + tr.R[i - 1])

    for i in range(tr.n):
        tr.R[i] = tr.junction_outR[i] * 0.999
        tr.L[i] = tr.junction_outL[i + 1] * 0.999
    tr.lip_output = tr.R[tr.n - 1]

    tr.nose_junc_outL[tr.nose_length] = tr.noseR[tr.nose_length - 1] * tr.lip_reflection

    for i in range(1, tr.nose_length):
        w = tr.nose_reflection[i] * (tr.noseR[i - 1] + tr.noseL[i])
        tr.nose_junc_outR[i] = tr.noseR[i - 1] - w
        tr.nose_junc_outL[i] = tr.noseL[i] + w

    for i in range(tr.nose_length):
        tr.noseR[i] = tr.nose_junc_outR[i]
        tr.noseL[i] = tr.nose_junc_outL[i + 1]
    tr.nose_output = tr.noseR[tr.nose_length - 1]

    return sp, tr


# int sp_voc_init(sp_data *sp, sp_voc *voc)
def sp_voc_init(sp: sp_data, voc: sp_voc) -> Tuple[sp_data, sp_voc]:
    voc.glot = glottis_init(voc.glot, sp.sr)  # /* initialize glottis */
    voc.tr = tract_init(sp, voc.tr)  # /* initialize vocal tract */
    voc.counter = 0
    return sp, voc


# int sp_voc_compute(sp_data *sp, sp_voc *voc, SPFLOAT *out)
def sp_voc_compute(sp: sp_data, voc: sp_voc, out: float) -> Tuple[sp_data, sp_voc, float]:
    vocal_output: float
    glot: float
    lmbd1: float
    lmbd2: float
    i: int

    if voc.counter == 0:
        voc.tr = tract_reshape(voc.tr)
        voc.tr = tract_calculate_reflections(voc.tr)
        for i in range(512):
            vocal_output = 0
            lmbd1 = float(i) / 512
            lmbd2 = float(i + 0.5) / 512
            sp, voc.glot, glot = glottis_compute(sp, voc.glot, lmbd1)

            sp, voc.tr = tract_compute(sp, voc.tr, glot, lmbd1)
            vocal_output += voc.tr.lip_output + voc.tr.nose_output

            sp, voc.tr = tract_compute(sp, voc.tr, glot, lmbd2)
            vocal_output += voc.tr.lip_output + voc.tr.nose_output
            voc.buf[i] = vocal_output * 0.125

    out = voc.buf[voc.counter]
    voc.counter = (voc.counter + 1) % 512
    return sp, voc, out


# int sp_voc_tract_compute(sp_data *sp, sp_voc *voc, SPFLOAT *in, SPFLOAT *out)
def sp_voc_tract_compute(sp: sp_data, voc: sp_voc, _in: float, out: float) -> Tuple[sp_data, sp_voc, float]:
    vocal_output: float
    lmbd1: float
    lmbd2: float

    if voc.counter == 0:
        voc.tr = tract_reshape(voc.tr)
        voc.tr = tract_calculate_reflections(voc.tr)

    vocal_output = 0
    lmbd1 = float(voc.counter) / 512
    lmbd2 = float(voc.counter + 0.5) / 512

    sp, voc.tr = tract_compute(sp, voc.tr, _in, lmbd1)
    vocal_output += voc.tr.lip_output + voc.tr.nose_output
    sp, voc.tr = tract_compute(sp, voc.tr, _in, lmbd2)
    vocal_output += voc.tr.lip_output + voc.tr.nose_output

    out = vocal_output * 0.125
    voc.counter = (voc.counter + 1) % 512

    return sp, voc, out


# void sp_voc_set_diameters(sp_voc *voc,
#     int blade_start,
#     int lip_start,
#     int tip_start,
#     SPFLOAT tongue_index,
#     SPFLOAT tongue_diameter,
#     SPFLOAT *diameters)
def sp_voc_set_diameters(voc: sp_voc,
                         blade_start: int,
                         lip_start: int,
                         tip_start: int,
                         tongue_index: float,
                         tongue_diameter: float,
                         diameters: List[float]) -> Tuple[sp_voc, List[float]]:
    # FIXME: NB Odd, voc is not used here That appears to be the case in the original code...

    i: int
    t: float
    fixed_tongue_diameter: float
    curve: float
    grid_offset: int = 0

    # for(i = blade_start i < lip_start i++) {
    for i in range(blade_start, lip_start):
        t = 1.1 * M_PI * float(tongue_index - i) / (tip_start - blade_start)
        fixed_tongue_diameter = 2 + (tongue_diameter - 2) / 1.5
        curve = (1.5 - fixed_tongue_diameter + grid_offset) * cos(t)
        if i == blade_start - 2 or i == lip_start - 1: curve *= 0.8
        if i == blade_start or i == lip_start - 2: curve *= 0.94
        diameters[i] = 1.5 - curve

    return voc, diameters


# void sp_voc_set_tongue_shape(sp_voc *voc,
#     SPFLOAT tongue_index,
#     SPFLOAT tongue_diameter)
def sp_voc_set_tongue_shape(voc: sp_voc,
                            tongue_index: float,
                            tongue_diameter: float) -> sp_voc:
    diameters: List[float]
    diameters = voc.tract_diameters
    voc, diameters = sp_voc_set_diameters(voc, 10, 39, 32,
                                          tongue_index, tongue_diameter, diameters);

    return voc
