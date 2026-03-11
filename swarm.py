import time
import threading

import numpy as np

from drone_state import DroneState

GRID_SIZE = 50
SCAN_RADIUS = 3
HEARTBEAT_INTERVAL = 0.5   # seconds between heartbeat broadcasts
HEARTBEAT_TIMEOUT = 2.0    # seconds before a follower considers the leader dead
STEP_SLEEP = 0.1           # seconds between simulation steps


class SwarmDrone:
    """
    A mission-capable drone with a leader-election state machine.

    Each drone maintains:
    - A 2-D position on the mission grid.
    - A DroneState (FOLLOWER / CANDIDATE / LEADER).
    - Heartbeat tracking so it can detect a silent leader.
    """

    def __init__(self, drone_id: int, grid_size: int = GRID_SIZE) -> None:
        self.drone_id = drone_id
        self.grid_size = grid_size
        self.state = DroneState.FOLLOWER
        self.is_alive = True
        self.leader_id: int = -1
        self.last_heartbeat: float = time.time()
        self.position = np.array(
            [np.random.randint(0, grid_size), np.random.randint(0, grid_size)]
        )

    # ------------------------------------------------------------------
    # Heartbeat helpers
    # ------------------------------------------------------------------

    def receive_heartbeat(self, leader_id: int) -> None:
        """Record a heartbeat from the leader."""
        self.leader_id = leader_id
        self.last_heartbeat = time.time()

    def is_heartbeat_timed_out(self) -> bool:
        """Return True if the leader's heartbeat has been silent too long."""
        return (time.time() - self.last_heartbeat) > HEARTBEAT_TIMEOUT

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_towards(self, target: np.ndarray) -> None:
        """Take one grid step towards *target*."""
        direction = target - self.position
        if np.any(direction != 0):
            step = np.sign(direction)
            self.position = np.clip(self.position + step, 0, self.grid_size - 1)

    def random_move(self) -> None:
        """Move one step in a random direction."""
        step = np.random.randint(-1, 2, size=2)
        self.position = np.clip(self.position + step, 0, self.grid_size - 1)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_cells(self, radius: int = SCAN_RADIUS) -> list:
        """Return all valid grid coordinates within *radius* of this drone."""
        x, y = self.position
        cells = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cells.append((int(nx), int(ny)))
        return cells


class Swarm:
    """
    Manages a swarm of drones with distributed leader election and 2-D sector mapping.

    Leader election uses the **Bully Algorithm**:
    1. Any follower that detects a leader timeout starts an election.
    2. It sends ELECTION messages to all drones with a higher ID.
    3. If a higher-ID drone is alive it takes over the election.
    4. The highest-ID alive drone wins and broadcasts VICTORY.

    The shared *grid* is a boolean numpy array that is progressively filled as
    drones scan their surrounding cells.
    """

    def __init__(self, num_drones: int = 50, grid_size: int = GRID_SIZE) -> None:
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=bool)
        self.drones: list[SwarmDrone] = [
            SwarmDrone(i, grid_size) for i in range(num_drones)
        ]
        self.event_log: list[str] = []
        self.running = False
        self._lock = threading.Lock()
        self._last_heartbeat_time: float = 0.0
        self._election_in_progress = False

        # Elect the initial leader (highest ID).
        self._elect_initial_leader()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def leader(self) -> SwarmDrone | None:
        """Return the current leader drone, or None if there is none."""
        for d in self.drones:
            if d.is_alive and d.state == DroneState.LEADER:
                return d
        return None

    @property
    def alive_count(self) -> int:
        return sum(1 for d in self.drones if d.is_alive)

    @property
    def scan_progress(self) -> float:
        """Fraction of grid cells that have been scanned (0.0 – 1.0)."""
        return float(np.sum(self.grid)) / (self.grid_size * self.grid_size)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.event_log.append(entry)
        print(entry)

    # ------------------------------------------------------------------
    # Leader election
    # ------------------------------------------------------------------

    def _elect_initial_leader(self) -> None:
        """Elect the initial leader as the drone with the highest ID."""
        alive = [d for d in self.drones if d.is_alive]
        if not alive:
            return
        leader = max(alive, key=lambda d: d.drone_id)
        leader.state = DroneState.LEADER
        for d in alive:
            if d is not leader:
                d.state = DroneState.FOLLOWER
                d.leader_id = leader.drone_id
        self._log(f"Drone {leader.drone_id} elected as initial leader")

    def _run_bully_election(self) -> None:
        """Execute the Bully Algorithm to elect a new leader."""
        if self._election_in_progress:
            return
        self._election_in_progress = True

        alive = [d for d in self.drones if d.is_alive]
        if not alive:
            self._log("No alive drones — mission failed")
            self.running = False
            self._election_in_progress = False
            return

        # Step 1: the follower with the lowest ID notices and starts the election.
        followers = [d for d in alive if d.state != DroneState.LEADER]
        if not followers:
            self._election_in_progress = False
            return

        initiator = min(followers, key=lambda d: d.drone_id)
        initiator.state = DroneState.CANDIDATE
        self._log(f"Election started by Drone {initiator.drone_id}")

        # Step 2: initiator sends ELECTION to drones with higher IDs.
        higher = [d for d in alive if d.drone_id > initiator.drone_id]
        if higher:
            # Higher-ID drones respond OK — the highest one wins.
            winner = max(higher, key=lambda d: d.drone_id)
            self._log(
                f"Drone {winner.drone_id} responded OK — taking over election"
            )
        else:
            # No higher drone replied — initiator wins.
            winner = initiator

        # Step 3: winner broadcasts VICTORY.
        for d in alive:
            d.state = DroneState.FOLLOWER
            d.leader_id = winner.drone_id
            d.last_heartbeat = time.time()  # Reset timeout clock.
        winner.state = DroneState.LEADER
        self._last_heartbeat_time = time.time()

        self._log(f"Drone {winner.drone_id} elected as new leader")
        self._log("Mission resumed")
        self._election_in_progress = False

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _broadcast_heartbeat(self) -> None:
        """Leader sends a heartbeat to all alive followers."""
        leader = self.leader
        if leader is None:
            return
        self._last_heartbeat_time = time.time()
        for d in self.drones:
            if d.is_alive and d is not leader:
                d.receive_heartbeat(leader.drone_id)

    def _check_heartbeat_timeout(self) -> None:
        """
        If there is no alive leader and any follower has timed out,
        trigger a Bully election.
        """
        if self.leader is not None:
            return
        if self._election_in_progress:
            return
        alive_followers = [
            d for d in self.drones if d.is_alive and d.state == DroneState.FOLLOWER
        ]
        if any(d.is_heartbeat_timed_out() for d in alive_followers):
            self._log("Leader heartbeat timeout detected")
            self._run_bully_election()

    # ------------------------------------------------------------------
    # Mission logic
    # ------------------------------------------------------------------

    def _move_drones(self) -> None:
        """Steer each alive drone towards the nearest unscanned grid cell."""
        unscanned = np.argwhere(~self.grid)
        if len(unscanned) == 0:
            return
        for drone in self.drones:
            if not drone.is_alive:
                continue
            distances = np.linalg.norm(unscanned - drone.position, axis=1)
            nearest_idx = int(np.argmin(distances))
            drone.move_towards(unscanned[nearest_idx])

    def _scan_grid(self) -> None:
        """Each alive drone scans the cells around its current position."""
        for drone in self.drones:
            if drone.is_alive:
                for cell in drone.scan_cells():
                    self.grid[cell] = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def kill_leader(self) -> None:
        """Destroy the current leader and trigger a new election."""
        leader = self.leader
        if leader is None:
            self._log("No active leader to kill")
            return
        leader.is_alive = False
        leader.state = DroneState.FOLLOWER
        self._log(f"Leader Drone {leader.drone_id} destroyed")
        self._run_bully_election()

    def step(self) -> None:
        """Advance the simulation by one step (thread-safe)."""
        if not self.running:
            return
        with self._lock:
            now = time.time()
            if now - self._last_heartbeat_time >= HEARTBEAT_INTERVAL:
                self._broadcast_heartbeat()

            self._check_heartbeat_timeout()
            self._move_drones()
            self._scan_grid()

            if self.scan_progress >= 1.0:
                self._log("Mission complete! All sectors scanned.")
                self.running = False

    def start(self) -> None:
        """Mark the swarm as running and log the event."""
        self.running = True
        self._log("Mission started")

    def stop(self) -> None:
        """Stop the simulation loop."""
        self.running = False
        self._log("Mission stopped")

    def export_map(self, filename: str = "mission_map.jpg") -> None:
        """Save the current mission map to *filename* as a JPEG image."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            self.grid.T,
            cmap="Blues",
            origin="lower",
            extent=[0, self.grid_size, 0, self.grid_size],
        )

        for drone in self.drones:
            if not drone.is_alive:
                continue
            x, y = drone.position
            if drone.state == DroneState.LEADER:
                ax.scatter(x, y, c="red", s=120, marker="*", zorder=5)
            elif drone.state == DroneState.CANDIDATE:
                ax.scatter(x, y, c="yellow", s=80, marker="^", zorder=4)
            else:
                ax.scatter(x, y, c="white", s=30, marker="o", zorder=3, alpha=0.8)

        coverage = self.scan_progress * 100
        ax.set_title(f"Mission Map — {coverage:.1f}% Scanned")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close(fig)
        self._log(f"Mission map exported to {filename}")
