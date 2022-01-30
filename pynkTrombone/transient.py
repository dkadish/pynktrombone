from .consts import MAX_TRANSIENTS
import numba
from numba.experimental import jitclass
from numba import int32, float64, boolean, int64
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
    """ Transient class
    Saving the transient state temporary.
    ある瞬間の状態を一時的に保持します。
    """
    def __init__(self,id:int) -> None:
        self.position: int = 0
        self.time_alive: float = 0
        self.lifetime: float
        self.strength: float = 0
        self.exponent: float = 0
        self.is_free: bool = True
        self.id: int = id

spec = OrderedDict()
inst_trans = Transient.class_type.instance_type
spec["pool"] = numba.types.List(inst_trans)
spec["free_ids"] = numba.types.List(int64)

@jitclass(spec)
class TransientPool:
    """ Transient Pool
    Transientを複数保持しておくためのプールオブジェクトです。
    Tract.computeの時に有効なTransientのリストを返します。
    
    """

    def __init__(self):
        """
        Prepair invalid transients.
        """
        self.pool: list[Transient] = [Transient(i) for i in range(MAX_TRANSIENTS)]
        fi = [*range(MAX_TRANSIENTS)]
        fi.reverse()
        self.free_ids = fi

    def append(self, position: int) -> None:
        """
        appending a transient of value of the position.
        If there is no free Transient, do nothing.

        positionの値のTransientを追加します。
        freeなTransientが存在しない場合は何もしません
        """
        free_id : int
        t: Transient

        if len(self.free_ids) == 0:
            return
        free_id = self.free_ids.pop()

        t = self.pool[free_id]
        t.is_free = False
        t.time_alive = 0
        t.lifetime = 0.2
        t.strength = 0.3
        t.exponent = 200
        t.position = position

    def remove(self, id: int) -> None:
        """
        Disable a valid transient and add its id to free_ids.
        有効なtransientを無効にし、free_idsにそのidを追加します。
        """
        t = self.pool[id]
        if not t.is_free:
            t.is_free = True
            self.free_ids.append(id)

    @property
    def size(self) -> int:
        return MAX_TRANSIENTS - len(self.free_ids)

    def get_valid_transients(self) -> list[Transient]:
        return [t for t in self.pool if not t.is_free]