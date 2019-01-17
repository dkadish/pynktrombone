def move_towards(current: float, target: float, amt_up: float, amt_down: float):
    if current < target:
        tmp = current + amt_up
        return min(tmp, target)
    else:
        tmp = current - amt_down
        return max(tmp, target)

#FIXME implement or refactor so that sp_data conforms to numpy/scipy audio format
class sp_data:
    pass