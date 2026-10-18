import numpy as np

RES = 1 / 32  # beat resolution of the tempo integration


class TempoMap:
    """Piecewise-linear master tempo; maps global beats <-> seconds."""

    def __init__(self, points: list[tuple[float, float]], total_beats: float):
        pts = sorted(points, key=lambda p: p[0])
        self.pb = np.array([p[0] for p in pts], float)
        self.pv = np.array([p[1] for p in pts], float)
        self.grid = np.arange(0, total_beats + 64, RES)
        bpm = np.interp(self.grid, self.pb, self.pv)
        mid = 0.5 * (bpm[1:] + bpm[:-1])
        self.times = np.concatenate([[0.0], np.cumsum(60.0 / mid * RES)])

    def bpm_at(self, beat):
        return np.interp(beat, self.pb, self.pv)

    def time(self, beat):
        return np.interp(beat, self.grid, self.times)

    def beat(self, t):
        return np.interp(t, self.times, self.grid)
