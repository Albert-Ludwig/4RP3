"""Time-aware extension of GP-Planning_reference/src/planners.py::grg_plan.

Preserves recursive waypoint selection, feasible budget splits, left/right
growth, and committed-prefix scoring. Route order replaces unordered node sets.
No fixed-set approximation guarantee is claimed for this temporal objective.
"""

from dataclasses import dataclass
import time

import numpy as np


def information_gain(cov, noise):
    if noise <= 0:
        raise ValueError("Observation variance must be positive")
    if cov.shape[-1] == 0:
        return np.zeros(cov.shape[:-2])
    sign, value = np.linalg.slogdet(np.eye(cov.shape[-1]) + cov / noise)
    if np.any(sign <= 0):
        raise FloatingPointError("Invalid information covariance")
    return value * 0.5


@dataclass
class Plan:
    route: list
    arrivals_s: list
    finish_s: float
    gain: float
    runtime_s: float
    evaluations: int
    gp_version: int


class GRGPlanner:
    def __init__(self, points, travel_s, start_h, sample_s, depth=2, time_splits=2,
                 home=0, allow_revisit=False):
        self.points = np.asarray(points)
        self.travel = np.asarray(travel_s)
        self.start_h, self.sample_s = start_h, sample_s
        self.depth, self.time_splits = depth, time_splits
        self.home, self.allow_revisit = home, allow_revisit
        if depth < 1 or time_splits < 1 or sample_s < 0:
            raise ValueError("Invalid search settings")

    def schedule(self, route, elapsed):
        arrivals, query = [], []
        for a, b in zip(route[:-1], route[1:]):
            elapsed += self.travel[a, b]
            if b != self.home:
                arrivals.append(float(elapsed))
                query.append([*self.points[b], self.start_h + elapsed / 3600])
                elapsed += self.sample_s
        return np.array(query).reshape(-1, 3), arrivals, float(elapsed)

    def score_routes(self, routes, gp, elapsed):
        scores = np.zeros(len(routes))
        groups = {}
        for i, route in enumerate(routes):
            query = self.schedule(route, elapsed)[0]
            if len(query):
                groups.setdefault(len(query), []).append((i, query))
        for group in groups.values():
            index, queries = zip(*group)
            scores[list(index)] = information_gain(gp.covariance_many(np.array(queries)), gp.noise)
        return scores

    def plan(self, gp, current, elapsed, remaining, visited, blocked=()):
        started = time.perf_counter()
        deadline = elapsed + remaining
        excluded = set(blocked) | (set() if self.allow_revisit else set(visited))
        available = [i for i in range(len(self.points)) if i not in excluded and i != self.home]
        cache = {}

        def scores(routes):
            missing = list(dict.fromkeys(tuple(p) for p in routes if tuple(p) not in cache))
            if missing:
                values = self.score_routes(missing, gp, elapsed)
                cache.update(zip(missing, values))
            return [cache[tuple(p)] for p in routes]

        def search(prefix, end, end_s, depth):
            start = prefix[-1]
            now = self.schedule(prefix, elapsed)[2]
            base = prefix if start == end else prefix + [end]
            if self.schedule(base, elapsed)[2] > end_s + 1e-8:
                return None
            if depth == 0:
                return base
            choices = [v for v in available if v not in prefix and v != end]
            if depth == 1:
                # At depth one, all feasible split deadlines produce the same
                # direct left/right route. Score each such route once, in a batch.
                routes = [base]
                for v in choices:
                    route = prefix + [v, end]
                    if self.schedule(route, elapsed)[2] <= end_s + 1e-8:
                        routes.append(route)
                return routes[int(np.argmax(scores(routes)))]
            best, best_score = base, scores([base])[0]
            for v in choices:
                lo = now + self.travel[start, v] + self.sample_s
                hi = end_s - self.travel[v, end] - (self.sample_s if end != self.home else 0)
                if lo > hi + 1e-8:
                    continue
                for split in np.unique(np.linspace(lo, hi, self.time_splits + 2)[1:-1]):
                    left = search(prefix, v, split, depth - 1)
                    if left is None:
                        continue
                    right = search(left, end, end_s, depth - 1)
                    if right is None:
                        continue
                    value = scores([right])[0]
                    if value > best_score + 1e-12:
                        best, best_score = right, value
            return best

        route = search([current], self.home, deadline, self.depth)
        if route is None:
            return Plan([], [], np.inf, 0.0, time.perf_counter() - started, len(cache), gp.version)
        _, arrivals, finish = self.schedule(route, elapsed)
        return Plan(route, arrivals, finish, float(scores([route])[0]),
                    time.perf_counter() - started, len(cache), gp.version)
