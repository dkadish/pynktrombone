MAX_TRANSIENTS = 4


class Transient:
    # unsigned int id
    # struct transient *next

    def __init__(self, position, id, next):
        self.position = position

        # Default parameters from voc.c#494
        self.time_alive = 0
        self.lifetime = 0.2
        self.strength = 0.3
        self.exponent = 200
        self.is_free = 0  # FIXME Should be bool
        self.id = id
        self.next = next


class TransientPool:
    # FIXME Should be possible to do this more efficiently with proper Python
    #  lists
    def __init__(self):
        self.pool = []
        self.root = None
        self.size = 0
        self.next_free = -1

    '''Appends an obstruction during the reshaping of the vocal tract.
    
    This is drawn from Paul Batchelor's documentation, p29.

    :param position:
    :return: 0 on failure, 1 on success
    '''
    def append(self, position):
        free_id = self.next_free

        # Check and see if the pool is full. If this is so, return 0
        if self.size == MAX_TRANSIENTS: return 0

        # If there is no recorded next free, search for the next free transient
        if free_id == -1:
            for i in range(MAX_TRANSIENTS):
                if self.pool[i].is_free:
                    free_id = i
                    break

        # If there is no free transient, return 0
        if free_id == -1: return 0

        # With a transient found, assign the current root of the list to be
        # the next value in the transient. (It does not matter if the root is
        # NULL, because the size of the list will prevent it from ever being
        # accessed.)
        transient = Transient(position, free_id, self.next)

        self.pool[free_id] = transient
        self.root = transient

        # Increase the size of the pool by 1.
        self.size += 1

        # Set the next free parameter to be −1.
        self.next_free = -1

        return 0  # FIXME Should this return 1?

    def remove(self, id):
        self.next_free = id
        n = self.root

        # If the transient *is* the root, set the root to be the next value.
        # Decrease the size by one, and return
        if id == n.id:
            self.root = n.next
            self.size -= 1
            return

        # Iterate through the list and search for the entry
        for i in range(self.size):
            if n.next.id == id:
                # Once the entry has been found, decrease the pool size by 1.
                self.size -= 1
                # The transient, now free for reuse, can now be toggled to be free,
                # and it can be the next variable ready to be used again
                n.next.is_free = 1
                n.next = n.next.next
                break

            n = n.next
