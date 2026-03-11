from enum import Enum, auto


class DroneState(Enum):
    """
    Represents the state of a drone in the leader election state machine.

    States:
    - FOLLOWER: Normal operating state; drone follows the current leader.
    - CANDIDATE: Transition state during an election; drone is campaigning to become leader.
    - LEADER: The drone is the current swarm leader responsible for broadcasting heartbeats.
    """

    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()
