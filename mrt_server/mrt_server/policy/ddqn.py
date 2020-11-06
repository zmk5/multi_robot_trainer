"""Double DQN model class for RL experiments with neural net function approx.

Written by: Zahi Kakish (zmk5)
"""
from mrt_server.policy.dqn import ServerPolicyDQN


class ServerPolicyDDQN(ServerPolicyDQN):
    """Double DQN Class containing all relevant RL information."""

    @property
    def atype(self):
        """Return type of RL algorithm as string."""
        return 'DDQN'
