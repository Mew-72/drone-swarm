"""
Tests for the self-healing drone swarm simulation.

Covers:
- DroneState enum
- SwarmDrone creation, heartbeat, scanning, movement
- Swarm initial leader election
- Simulation stepping and map coverage
- Kill-leader + Bully-algorithm re-election
- Event logging
- Drone.is_alive / Drone.state attributes
- Map export
- Graceful handling of kill with no active leader
"""

import os
import time
import tempfile

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before any other matplotlib import

from drone_state import DroneState
from swarm import (
    Swarm,
    SwarmDrone,
    GRID_SIZE,
    SCAN_RADIUS,
    HEARTBEAT_TIMEOUT,
)
from drone import Drone


# ---------------------------------------------------------------------------
# DroneState
# ---------------------------------------------------------------------------

class TestDroneState:
    def test_has_follower(self):
        assert DroneState.FOLLOWER.name == "FOLLOWER"

    def test_has_candidate(self):
        assert DroneState.CANDIDATE.name == "CANDIDATE"

    def test_has_leader(self):
        assert DroneState.LEADER.name == "LEADER"

    def test_all_three_values_are_distinct(self):
        states = list(DroneState)
        assert len(states) == 3
        assert len(set(states)) == 3


# ---------------------------------------------------------------------------
# SwarmDrone
# ---------------------------------------------------------------------------

class TestSwarmDrone:
    def test_creation_defaults(self):
        d = SwarmDrone(drone_id=7, grid_size=50)
        assert d.drone_id == 7
        assert d.state is DroneState.FOLLOWER
        assert d.is_alive is True
        assert d.leader_id == -1

    def test_position_within_grid(self):
        d = SwarmDrone(drone_id=0, grid_size=30)
        assert 0 <= d.position[0] < 30
        assert 0 <= d.position[1] < 30

    def test_heartbeat_not_timed_out_immediately(self):
        d = SwarmDrone(drone_id=1)
        assert not d.is_heartbeat_timed_out()

    def test_heartbeat_timed_out_after_delay(self):
        d = SwarmDrone(drone_id=2)
        d.last_heartbeat = time.time() - (HEARTBEAT_TIMEOUT + 1)
        assert d.is_heartbeat_timed_out()

    def test_receive_heartbeat_resets_timeout(self):
        d = SwarmDrone(drone_id=3)
        d.last_heartbeat = time.time() - (HEARTBEAT_TIMEOUT + 1)
        assert d.is_heartbeat_timed_out()
        d.receive_heartbeat(leader_id=9)
        assert not d.is_heartbeat_timed_out()
        assert d.leader_id == 9

    def test_scan_cells_returns_valid_coords(self):
        d = SwarmDrone(drone_id=0, grid_size=50)
        cells = d.scan_cells(radius=SCAN_RADIUS)
        assert len(cells) > 0
        for x, y in cells:
            assert 0 <= x < 50
            assert 0 <= y < 50

    def test_scan_cells_count(self):
        d = SwarmDrone(drone_id=0, grid_size=50)
        # Force drone to center so no edge clipping
        d.position = np.array([25, 25])
        cells = d.scan_cells(radius=2)
        assert len(cells) == (2 * 2 + 1) ** 2  # 25 cells

    def test_move_towards_changes_position(self):
        d = SwarmDrone(drone_id=0, grid_size=50)
        d.position = np.array([10, 10])
        target = np.array([20, 20])
        d.move_towards(target)
        # Should have stepped one cell closer in each axis
        assert d.position[0] == 11
        assert d.position[1] == 11

    def test_move_towards_stays_in_bounds(self):
        d = SwarmDrone(drone_id=0, grid_size=10)
        d.position = np.array([9, 9])
        d.move_towards(np.array([100, 100]))
        assert 0 <= d.position[0] < 10
        assert 0 <= d.position[1] < 10

    def test_random_move_stays_in_bounds(self):
        d = SwarmDrone(drone_id=0, grid_size=10)
        for _ in range(50):
            d.random_move()
        assert 0 <= d.position[0] < 10
        assert 0 <= d.position[1] < 10


# ---------------------------------------------------------------------------
# Swarm — initial state
# ---------------------------------------------------------------------------

class TestSwarmInitial:
    def test_alive_count_equals_num_drones(self):
        s = Swarm(num_drones=8, grid_size=20)
        assert s.alive_count == 8

    def test_initial_leader_is_highest_id(self):
        s = Swarm(num_drones=10, grid_size=20)
        assert s.leader is not None
        assert s.leader.drone_id == 9

    def test_initial_leader_state(self):
        s = Swarm(num_drones=5, grid_size=20)
        assert s.leader.state is DroneState.LEADER

    def test_all_other_drones_are_followers(self):
        s = Swarm(num_drones=5, grid_size=20)
        for d in s.drones:
            if d is not s.leader:
                assert d.state is DroneState.FOLLOWER

    def test_initial_scan_progress_is_zero(self):
        s = Swarm(num_drones=5, grid_size=20)
        assert s.scan_progress == 0.0

    def test_grid_shape(self):
        s = Swarm(num_drones=5, grid_size=30)
        assert s.grid.shape == (30, 30)


# ---------------------------------------------------------------------------
# Swarm — simulation step
# ---------------------------------------------------------------------------

class TestSwarmStep:
    def test_step_increases_coverage(self):
        s = Swarm(num_drones=10, grid_size=20)
        s.start()
        for _ in range(5):
            s.step()
        assert s.scan_progress > 0.0

    def test_step_noop_when_not_running(self):
        s = Swarm(num_drones=5, grid_size=20)
        # Do NOT call start() — running remains False
        s.step()
        assert s.scan_progress == 0.0

    def test_mission_completes_eventually(self):
        s = Swarm(num_drones=20, grid_size=10)
        s.start()
        for _ in range(500):
            if not s.running:
                break
            s.step()
        assert s.scan_progress == 1.0
        assert not s.running
        assert any("Mission complete" in e for e in s.event_log)


# ---------------------------------------------------------------------------
# Swarm — leader election (Bully algorithm)
# ---------------------------------------------------------------------------

class TestSwarmElection:
    def test_kill_leader_removes_leader(self):
        s = Swarm(num_drones=5, grid_size=20)
        s.start()
        old_id = s.leader.drone_id
        s.kill_leader()
        # Old leader must now be dead
        assert not s.drones[old_id].is_alive

    def test_new_leader_elected_after_kill(self):
        s = Swarm(num_drones=5, grid_size=20)
        s.start()
        old_id = s.leader.drone_id
        s.kill_leader()
        assert s.leader is not None
        assert s.leader.drone_id != old_id

    def test_new_leader_has_lower_id_than_old(self):
        s = Swarm(num_drones=5, grid_size=20)
        s.start()
        old_id = s.leader.drone_id
        s.kill_leader()
        # Bully: new leader is the next highest alive ID
        assert s.leader.drone_id < old_id

    def test_election_log_contains_key_events(self):
        s = Swarm(num_drones=5, grid_size=20)
        s.start()
        s.kill_leader()
        log = "\n".join(s.event_log)
        assert "destroyed" in log
        assert "Election started" in log
        assert "elected as new leader" in log
        assert "Mission resumed" in log

    def test_multiple_successive_elections(self):
        s = Swarm(num_drones=5, grid_size=20)
        s.start()
        for _ in range(4):  # kill 4 of 5 drones
            if s.leader:
                s.kill_leader()
        # One drone still alive and acting as leader
        assert s.alive_count == 1
        assert s.leader is not None

    def test_kill_all_drones_stops_mission(self):
        s = Swarm(num_drones=2, grid_size=10)
        s.start()
        s.kill_leader()  # kills drone 1
        s.kill_leader()  # kills drone 0
        assert s.leader is None
        assert not s.running

    def test_kill_with_no_active_leader_logs_gracefully(self):
        s = Swarm(num_drones=2, grid_size=10)
        s.start()
        s.kill_leader()
        s.kill_leader()  # kills the last drone → no leader
        s.kill_leader()  # nothing to kill
        assert any("No active leader" in e for e in s.event_log)

    def test_heartbeat_timeout_triggers_election(self):
        s = Swarm(num_drones=4, grid_size=20)
        s.start()
        # Simulate leader death without calling kill_leader
        leader = s.leader
        leader.is_alive = False
        leader.state = DroneState.FOLLOWER
        # Age all follower heartbeats past the timeout threshold
        for d in s.drones:
            if d.is_alive:
                d.last_heartbeat = time.time() - (HEARTBEAT_TIMEOUT + 1)
        # A single step should detect the timeout and run an election
        s.step()
        assert s.leader is not None
        assert s.leader.drone_id != leader.drone_id


# ---------------------------------------------------------------------------
# Drone (formation simulation) — backward compatibility
# ---------------------------------------------------------------------------

class TestDroneCompat:
    def test_is_alive_default(self):
        d = Drone(np.array([1.0, 2.0, 3.0]), index=0)
        assert d.is_alive is True

    def test_state_default(self):
        d = Drone(np.array([0.0, 0.0, 0.0]), index=5)
        assert d.state is DroneState.FOLLOWER

    def test_existing_api_unchanged(self):
        d = Drone(np.array([1.0, 2.0, 3.0]), index=0)
        np.testing.assert_array_equal(d.get_position(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(d.communicate(), [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Map export
# ---------------------------------------------------------------------------

class TestMapExport:
    def test_export_creates_file(self):
        s = Swarm(num_drones=5, grid_size=10)
        s.start()
        for _ in range(20):
            s.step()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
        try:
            s.export_map(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_export_logs_event(self):
        s = Swarm(num_drones=3, grid_size=10)
        s.start()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
        try:
            s.export_map(path)
            assert any("exported" in e for e in s.event_log)
        finally:
            os.unlink(path)
