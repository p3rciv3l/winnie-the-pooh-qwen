"""Min-heap priority queue tracker for top-k activating examples per neuron."""

import heapq
from dataclasses import dataclass


@dataclass
class ActivationRecord:
    """A single activation record for a neuron."""
    activation: float
    text: str  # The full text context
    token_idx: int  # Position of the activating token in the text
    token: str  # The specific token that activated
    shard_id: str | None = None  # Source shard for traceability

    def __lt__(self, other: "ActivationRecord") -> bool:
        return self.activation < other.activation

    def __eq__(self, other: "ActivationRecord") -> bool:
        return self.activation == other.activation

    def to_dict(self) -> dict:
        return {
            "activation": self.activation,
            "text": self.text,
            "token_idx": self.token_idx,
            "token": self.token,
            "shard_id": self.shard_id,
        }


class NeuronHeapTracker:
    """Maintains top-k activating examples for each neuron using min-heaps."""

    def __init__(self, neuron_ids: list[str], top_k: int = 50):
        self.top_k = top_k
        self.heaps: dict[str, list[ActivationRecord]] = {n: [] for n in neuron_ids}

    def update(
        self,
        neuron_id: str,
        activation: float,
        text: str,
        token_idx: int,
        token: str,
        shard_id: str | None = None,
    ) -> bool:
        if neuron_id not in self.heaps:
            return False

        heap = self.heaps[neuron_id]
        record = ActivationRecord(
            activation=activation,
            text=text,
            token_idx=token_idx,
            token=token,
            shard_id=shard_id,
        )

        if len(heap) < self.top_k:
            heapq.heappush(heap, record)
            return True
        elif activation > heap[0].activation:
            heapq.heapreplace(heap, record)
            return True
        return False

    def get_top_k(self, neuron_id: str) -> list[ActivationRecord]:
        """Get top-k records for a neuron, sorted by activation (highest first)."""
        if neuron_id not in self.heaps:
            return []
        return sorted(self.heaps[neuron_id], key=lambda r: r.activation, reverse=True)

    def get_min_activation(self, neuron_id: str) -> float | None:
        """Get the minimum activation in the heap (threshold for new entries)."""
        if neuron_id not in self.heaps or not self.heaps[neuron_id]:
            return None
        return self.heaps[neuron_id][0].activation
