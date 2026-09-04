from typing import NamedTuple, Optional

from ..utils.typing import Array
from ..utils.typing import Action, Reward, Cost, Done
from ..utils.graph import GraphsTuple


class Rollout(NamedTuple):
    graph: GraphsTuple
    actions: Action
    rnn_states: Array
    rewards: Reward
    costs: Cost
    dones: Done
    log_pis: Optional[Array]
    next_graph: GraphsTuple

    @property
    def length(self) -> int:
        return self.rewards.shape[0]

    @property
    def time_horizon(self) -> int:
        return self.rewards.shape[1]

    @property
    def num_agents(self) -> int:
        return self.rewards.shape[2]

    @property
    def n_data(self) -> int:
        return self.length * self.time_horizon


class SafetyBatch(NamedTuple):
    """Off-policy transitions used only to pretrain the Graph-HJ critic."""

    graph: GraphsTuple
    actions: Action
    constraints: Array
    next_graph: GraphsTuple
    next_constraints: Array
    dones: Done


class GCBFTransitionBatch(NamedTuple):
    """Transitions and horizon labels consumed by GCBF+ updates."""

    graph: GraphsTuple
    next_graph: GraphsTuple
    safe_mask: Array
    unsafe_mask: Array
