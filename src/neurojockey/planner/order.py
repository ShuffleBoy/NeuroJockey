"""Track ordering: beam search over tempo / key compatibility and a target energy curve."""

import numpy as np

from .compat import pair_cost


def energy_target(x: float, curve: str) -> float:
    if curve == "rise":
        return x
    if curve == "flat":
        return 0.5
    # "peak": warm up, peak at ~75 %, cool down a bit at the end
    return x / 0.75 if x <= 0.75 else 1 - 0.4 * (x - 0.75) / 0.25


def order_tracks(tracks: list, curve: str = "peak", max_shift: int = 1, beam: int = 64) -> list[int]:
    n = len(tracks)
    if n <= 2:
        return sorted(range(n), key=lambda i: tracks[i].energy_level)
    levels = np.array([t.energy_level for t in tracks])
    rank = levels.argsort().argsort() / (n - 1)
    pc = [[pair_cost(tracks[i], tracks[j], max_shift) if i != j else 0 for j in range(n)] for i in range(n)]

    def e_cost(i, pos):
        return 1.5 * abs(rank[i] - energy_target(pos / (n - 1), curve))

    states = [(e_cost(i, 0), [i]) for i in range(n)]
    for pos in range(1, n):
        nxt = []
        for cost, path in states:
            used = set(path)
            for j in range(n):
                if j not in used:
                    # build in steps: neighbours more than ~1 energy level (of 10) apart lose the crowd
                    step = 3.0 * max(0.0, abs(levels[j] - levels[path[-1]]) - 0.1)
                    nxt.append((cost + pc[path[-1]][j] + e_cost(j, pos) + step, path + [j]))
        nxt.sort(key=lambda s: s[0])
        states = nxt[:beam]
    return states[0][1]
