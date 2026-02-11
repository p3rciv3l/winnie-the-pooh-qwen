"""Min-heap priority queue tracker for top-k activating examples per neuron."""

import heapq
from dataclasses import dataclass


@dataclass
class ActivationRecord:
    """A single activation record for a neuron."""
    activation: float
    token_activations: list[tuple[int, float]]
    shard_id: str
    row_idx: int

    def __lt__(self, other: "ActivationRecord") -> bool:
        return self.activation < other.activation

    def to_dict(self) -> dict:
        return {
            "activation": self.activation,
            "token_activations": self.token_activations,
            "shard_id": self.shard_id,
            "row_idx": self.row_idx
        }


class NeuronHeapTracker:
    """Maintains top-k and bottom-k activating examples for each neuron using heaps."""

    def __init__(self, neuron_ids: list[str], k: int = 5000):
        self.k = k
        self.minimums: dict[str, list[ActivationRecord]] = {n: [] for n in neuron_ids}
        self.maximums: dict[str, list[ActivationRecord]] = {n: [] for n in neuron_ids}

    def update(
        self,
        neuron_id: str,
        max_activation: float,
        token_activations: list[tuple[int, float]],
        shard_id: str,
        row_idx: int,
    ) -> bool:

        mins = self.minimums[neuron_id]
        maxs = self.maximums[neuron_id]
        record = ActivationRecord(
            activation=max_activation,
            token_activations=token_activations,
            shard_id=shard_id,
            row_idx=row_idx,
        )
        
        if len(mins) < self.k: 
            heapq.heappush(mins, (-1 * record.activation, record))
        elif (-1 * record.activation) > mins[0][0]: 
            heapq.heapreplace(mins, (-1 * record.activation, record))

        if len(maxs) < self.k:
            heapq.heappush(maxs, (record.activation, record))
        elif record.activation > maxs[0][0]:
            heapq.heapreplace(maxs, (record.activation, record))

    def get_top_and_bottom_k(self, neuron_id: str) -> tuple[list[ActivationRecord], list[ActivationRecord]]:
        """Get top-k records for a neuron, sorted by activation (highest first)."""
        
        top_k_tuples = sorted(self.maximums[neuron_id], key=lambda rt: rt[0], reverse=True)
        bottom_k_tuples = sorted(self.minimums[neuron_id], key=lambda rt: rt[0])

        top_k = [el[1] for el in top_k_tuples]
        bottom_k = [el[1] for el in bottom_k_tuples]

        return bottom_k, top_k
