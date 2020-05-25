from itertools import count
from typing import List

from pynkTrombone import voc


class Transient:
    ids = count(0)

    def __init__(self):
        self.position: int = 0
        self.time_alive: float = 0
        self.lifetime: float
        self.strength: float = 0
        self.exponent: float = 0
        self.is_free: str = 1
        self.id: int = next(self.ids)
        self.next: Transient  # PTR

    @classmethod
    def reset_count(self):
        self.ids = count(0)


class TransientPool:
    def __init__(self):
        Transient.reset_count()

        self.pool: List[Transient] = []  # Should be limited to MAX_TRANSIENTS
        self.root: Transient = None  # PTR
        self.size: int = 0
        self.next_free: int = 0

        for i in range(voc.MAX_TRANSIENTS):
            self.pool.append(Transient())

    # static int append_transient(TransientPool *self, int position)
    # CHANGE: self is not a pointer, is returned from fn
    def append(self, position: int) -> None:
        i: int
        free_id: int
        # Transient *t
        t: Transient

        free_id = self.next_free
        if self.size == voc.MAX_TRANSIENTS:
            return

        if free_id == -1:
            for i in range(voc.MAX_TRANSIENTS):
                if self.pool[i].is_free:
                    free_id = i
                    break

        if free_id == -1:
            return

        t = self.pool[free_id]
        t.next = self.root
        self.root = t
        self.size += 1
        t.is_free = 0
        t.time_alive = 0
        t.lifetime = 0.2
        t.strength = 0.3
        t.exponent = 200
        t.position = position
        self.next_free = -1

    # static void remove_transient(TransientPool *self, unsigned int id)
    # CHANGE: self is not a pointer, is returned from fn
    def remove(self, id: int) -> None:
        i: int
        n: Transient

        self.next_free = id
        n = self.root
        if id == n.id:
            self.root = n.next
            self.size -= 1
            return

        for i in range(self.size):
            if n.next.id == id:
                self.size -= 1
                n.next.is_free = 1
                n.next = n.next.next
                break
            n = n.next
