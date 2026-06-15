"""Topology graph primitives.

Higher-level evaluator and oracle modules are intentionally imported from their
own modules to keep the Stage 2 skeleton acyclic.
"""

from .candidate_graph import CandidateEdge, CandidateGraph, canonical_edge_id

__all__ = [
    "CandidateEdge",
    "CandidateGraph",
    "canonical_edge_id",
]
