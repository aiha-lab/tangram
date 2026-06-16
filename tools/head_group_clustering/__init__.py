"""Head-group clustering tools.

Build a cluster map -- an assignment of every (layer, head) to a (cluster,
column) so that KV heads with similar compression-retention budget share a page
group, eliminating the max-pool over-allocation of adjacency-based head groups.
"""
from tools.head_group_clustering.clustering import (
    ClusterMap,
    build_clusters,
    over_allocation_stats,
)

__all__ = ["ClusterMap", "build_clusters", "over_allocation_stats"]
