"""Receding-horizon mission control through navigation and sensor contracts."""

from dataclasses import asdict
import numpy as np

from grg import information_gain
from interfaces import Navigation, TemperatureSource


class MissionController:
    def __init__(self, gp, planner, nav: Navigation, sensor: TemperatureSource,
                 config, observer=None):
        self.gp, self.planner, self.nav, self.sensor = gp, planner, nav, sensor
        self.config, self.observer = config, observer
        self.online, self.log, self.plans = [], [], []
        self.distance, self.travel_s, self.total_gain = 0.0, 0.0, 0.0
        self.stop_reason = "not_started"

    def run(self):
        c = self.config
        current, visited, blocked = 0, set(), set()
        if self.observer:
            self.observer(self.gp, self.nav.get_robot_state().elapsed_s, 0)
        for iteration in range(c["max_iterations"]):
            state = self.nav.get_robot_state()
            remaining = c["budget_s"] - state.elapsed_s
            plan = self.planner.plan(self.gp, current, state.elapsed_s, remaining, visited, blocked)
            self.plans.append({"iteration": iteration, "elapsed_s": state.elapsed_s, **asdict(plan)})
            row = {"iteration": iteration, "x_m": state.x_m, "y_m": state.y_m,
                   "elapsed_s": state.elapsed_s, "remaining_s": remaining,
                   "candidates": [i for i in range(len(self.planner.points))
                                  if i != 0 and i not in blocked
                                  and (self.planner.allow_revisit or i not in visited)],
                   "route": plan.route, "route_gain_nats": plan.gain,
                   "plan_s": plan.runtime_s, "gp_version": self.gp.version,
                   "gp_updated": False}
            if not plan.route or len(plan.route) < 3:
                self.stop_reason = "no_feasible_sampling_route"
                row["status"] = self.stop_reason
                self.log.append(row)
                break
            if plan.finish_s > c["budget_s"] + 1e-8:
                self.stop_reason = "infeasible_plan"
                row["status"] = self.stop_reason
                self.log.append(row)
                break
            node = plan.route[1]
            goal = self.planner.points[node]
            estimate = self.nav.estimate_travel_cost(state.xy, goal)
            back = self.nav.estimate_travel_cost(goal, self.planner.points[0])
            row.update(next_node=node, estimated_travel_s=estimate)
            if not np.isfinite(estimate + back) or estimate + c["sample_s"] + back > remaining + 1e-8:
                self.stop_reason = "return_reserve"
                row["status"] = self.stop_reason
                self.log.append(row)
                break
            result = self.nav.navigate_to(goal)
            actual = self.nav.get_robot_state()
            self.distance += result.distance_m
            self.travel_s += result.travel_s
            row.update(navigation_status=result.status, actual_travel_s=result.travel_s,
                       arrival_x_m=actual.x_m, arrival_y_m=actual.y_m,
                       arrival_s=actual.elapsed_s, arrival_t_h=c["start_h"] + actual.elapsed_s / 3600)
            if result.status != "arrived" or not np.allclose(actual.xy, goal, atol=1e-8, rtol=0):
                row["status"] = "navigation_failed"
                self.log.append(row)
                blocked.add(node)
                if not np.allclose(actual.xy, state.xy, atol=1e-8, rtol=0):
                    self.stop_reason = "navigation_failed_after_motion"
                    break
                continue
            current = node
            back = self.nav.estimate_travel_cost(actual.xy, self.planner.points[0])
            if actual.elapsed_s + c["sample_s"] + back > c["budget_s"] + 1e-8:
                row["status"] = "arrival_return_reserve"
                self.log.append(row)
                self.stop_reason = row["status"]
                break
            # Measurement is instantaneous at arrival; dwell delays departure.
            record = self.sensor.measure(actual.xy, row["arrival_t_h"])
            self.nav.wait(c["sample_s"])
            matches = (record.valid() and np.allclose([record.x_m, record.y_m], actual.xy,
                                                     atol=1e-8, rtol=0)
                       and abs(record.t_h - row["arrival_t_h"]) < 1e-10)
            if not matches:
                row["status"] = "invalid_measurement"
                blocked.add(node)
                self.log.append(row)
                continue
            query = np.array([[record.x_m, record.y_m, record.t_h]])
            gain = float(information_gain(self.gp.joint(query)[1], self.gp.noise))
            self.gp.append(record)
            self.online.append(record)
            self.total_gain += gain
            visited.add(node)
            row.update(status="sampled", temperature_c=record.temperature_c,
                       measurement_source=record.source, measurement_status=record.status,
                       gp_updated=True, gain_nats=gain, total_gain_nats=self.total_gain,
                       after_sample_s=self.nav.get_robot_state().elapsed_s)
            if self.observer:
                row.update(self.observer(self.gp, self.nav.get_robot_state().elapsed_s, len(self.online)))
            self.log.append(row)
        else:
            self.stop_reason = "iteration_limit"
        state = self.nav.get_robot_state()
        back = self.nav.estimate_travel_cost(state.xy, self.planner.points[0])
        if np.isfinite(back) and state.elapsed_s + back <= c["budget_s"] + 1e-8:
            result = self.nav.navigate_to(self.planner.points[0])
            self.distance += result.distance_m
            self.travel_s += result.travel_s
            return_status = result.status
        else:
            return_status = "no_feasible_return"
        final = self.nav.get_robot_state()
        return {"measurements": len(self.online), "planning_calls": len(self.plans),
                "travel_distance_m": self.distance, "travel_time_s": self.travel_s,
                "elapsed_s": final.elapsed_s, "budget_s": c["budget_s"],
                "returned_home": bool(return_status == "arrived"
                                      and np.allclose(final.xy, self.planner.points[0])),
                "return_status": return_status, "stop_reason": self.stop_reason,
                "information_gain_nats": self.total_gain,
                "grg_runtime_s": sum(p["runtime_s"] for p in self.plans)}
