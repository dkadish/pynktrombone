from .consts import MAX_TRANSIENTS

from numba.experimental import jitclass
from numba import int32, float64, boolean
from collections import OrderedDict

spec = OrderedDict()
spec["position"] = int32
spec["time_alive"] = float64
spec["lifetime"] = float64
spec["strength"] = float64
spec["exponent"] = float64
spec["is_free"] = boolean
spec["id"] = int32

@jitclass(spec)
class Transient:
    def __init__(self,id:int) -> None:
        self.position: int = 0
        self.time_alive: float = 0
        self.lifetime: float
        self.strength: float = 0
        self.exponent: float = 0
        self.is_free: bool = True
        self.id: int = id