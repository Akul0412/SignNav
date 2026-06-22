"""
Controller: executes the high-level Decision on the robot.

Placeholder for the OmniVLA prompt->action mapping (not ready yet). For now it
prints the action and holds the 'current action' so the loop can 'continue what
it was doing' when nothing new is detected. Real robot hooks (ROS cmd_vel, etc.)
go where marked.
"""

from .types import ActionType, Decision


class Controller:
    def __init__(self):
        self.current_action = ActionType.FORWARD
        self._log = []

    def execute(self, decision: Decision):
        self.current_action = decision.action
        self._emit(decision.action, decision.rationale)

    def continue_previous(self):
        self._emit(self.current_action, "(nothing new detected — continuing previous action)")

    def _emit(self, action: ActionType, why: str):
        line = f"[Controller] ACTION={action.value:12s} | {why}"
        print(line)
        self._log.append(line)
        # --- REAL ROBOT HOOK (fill in for Jetson + ROS) ---
        # e.g. publish Twist to /cmd_vel based on `action`:
        #   FORWARD     -> linear.x = +v
        #   TURN_LEFT   -> angular.z = +w
        #   TURN_RIGHT  -> angular.z = -w
        #   STOP/REROUTE-> linear.x = 0 (+ recovery behavior)
        # ---------------------------------------------------

    def history(self):
        return self._log