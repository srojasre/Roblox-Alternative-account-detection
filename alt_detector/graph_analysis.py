"""Graph-based relationship analysis for candidate Roblox accounts (Layer D)."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx

from alt_detector.schemas import AccountObservation

# Default free Rthro/standard animation asset IDs that carry low discriminative signal
COMMON_DEFAULT_AVATAR_ASSETS: set[int] = {
    2510230574,
    2510233257,
    2510235063,
    2510236649,
    2510238627,
    2510240941,
    2510242378,
    607702162,
    607785314,
}


class CandidateGraphAnalyzer:
    """Constructs a typed, rarity-weighted multi-entity graph to evaluate structural proximity.

    Graph proximity alone does not prove common ownership; these metrics serve as
    one bounded family of structural evidence alongside profile and group features.
    """

    def __init__(self, max_group_size_for_graph: int = 100_000) -> None:
        self.max_group_size_for_graph = max_group_size_for_graph

    def build_and_analyze(
        self,
        target: AccountObservation,
        candidates: list[AccountObservation],
    ) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
        """Build the candidate relationship graph and return per-candidate graph metrics."""
        graph = nx.Graph()
        all_accounts = [target] + [c for c in candidates if c.user_id != target.user_id]
        observed_uids = {acc.user_id for acc in all_accounts}

        target_node = f"user:{target.user_id}"
        for acc in all_accounts:
            unode = f"user:{acc.user_id}"
            graph.add_node(
                unode,
                node_type="user",
                user_id=acc.user_id,
                username=acc.username,
                is_target=(acc.user_id == target.user_id),
            )

        # Add group membership edges weighted by inverse log popularity and role rank
        for acc in all_accounts:
            unode = f"user:{acc.user_id}"
            if acc.groups:
                for grp in acc.groups:
                    mc = grp.member_count or 50_000
                    if mc > self.max_group_size_for_graph and grp.role_rank <= 1:
                        continue
                    gnode = f"group:{grp.group_id}"
                    if not graph.has_node(gnode):
                        graph.add_node(
                            gnode,
                            node_type="group",
                            group_id=grp.group_id,
                            member_count=mc,
                        )
                    rarity_w = 1.0 / math.log(max(mc, 2) + 1.0)
                    rank_mult = 1.0 + 0.65 * (max(0, min(grp.role_rank, 255)) / 255.0)
                    graph.add_edge(
                        unode,
                        gnode,
                        edge_type="group_membership",
                        weight=round(rarity_w * rank_mult, 5),
                    )
                    # Also connect shared specific role node in small/medium groups
                    if mc <= 5000 and grp.role_rank > 1:
                        rnode = f"role:{grp.group_id}:{grp.role_id}"
                        if not graph.has_node(rnode):
                            graph.add_node(rnode, node_type="group_role", member_count=max(2, mc // 10))
                        graph.add_edge(
                            unode,
                            rnode,
                            edge_type="elevated_role",
                            weight=round(rarity_w * 1.35, 5),
                        )

            # Add friendship edges (direct between observed users + shared friend intermediaries)
            if acc.friends:
                for fid in acc.friends:
                    if fid in observed_uids and fid != acc.user_id:
                        f_unode = f"user:{fid}"
                        graph.add_edge(
                            unode,
                            f_unode,
                            edge_type="direct_friend",
                            weight=0.75,
                        )
                    else:
                        fnode = f"friend:{fid}"
                        if not graph.has_node(fnode):
                            graph.add_node(fnode, node_type="friend_contact", user_id=fid)
                        graph.add_edge(
                            unode,
                            fnode,
                            edge_type="mutual_friend_link",
                            weight=0.30,
                        )

            # Add non-default avatar asset edges
            if acc.avatar and acc.avatar.asset_ids:
                for aid in acc.avatar.asset_ids:
                    if aid in COMMON_DEFAULT_AVATAR_ASSETS:
                        continue
                    anode = f"asset:{aid}"
                    if not graph.has_node(anode):
                        graph.add_node(anode, node_type="avatar_asset", asset_id=aid)
                    graph.add_edge(
                        unode,
                        anode,
                        edge_type="wears_asset",
                        weight=0.22,
                    )

            # Add favorite game edges
            if acc.favorite_games:
                for fav in acc.favorite_games:
                    gm_node = f"game:{fav.game_id}"
                    if not graph.has_node(gm_node):
                        graph.add_node(gm_node, node_type="game", game_id=fav.game_id)
                    graph.add_edge(
                        unode,
                        gm_node,
                        edge_type="favorite_game",
                        weight=0.25,
                    )

        # Prune singleton intermediate nodes (degree == 1) that do not bridge accounts
        to_prune = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("node_type") != "user" and graph.degree(n) <= 1
        ]
        graph.remove_nodes_from(to_prune)

        # Personalized PageRank rooted at target
        ppr_scores: dict[str, float] = {}
        if graph.number_of_edges() > 0 and target_node in graph:
            try:
                ppr_scores = nx.pagerank(
                    graph,
                    alpha=0.85,
                    personalization={target_node: 1.0},
                    weight="weight",
                    max_iter=100,
                )
            except Exception:
                ppr_scores = {}

        candidate_pprs = [
            ppr_scores.get(f"user:{c.user_id}", 0.0)
            for c in candidates
            if c.user_id != target.user_id
        ]
        max_cand_ppr = max(candidate_pprs) if candidate_pprs else 0.0

        # Community detection
        target_community: set[str] = set()
        num_communities = 0
        if graph.number_of_edges() > 0:
            try:
                communities = list(
                    nx.community.greedy_modularity_communities(graph, weight="weight")
                )
                num_communities = len(communities)
                for comm in communities:
                    if target_node in comm:
                        target_community = set(comm)
                        break
            except Exception:
                target_community = set()

        target_neighbors = set(graph.neighbors(target_node)) if target_node in graph else set()
        metrics_by_uid: dict[int, dict[str, float]] = {}

        for cand in candidates:
            if cand.user_id == target.user_id:
                continue
            cnode = f"user:{cand.user_id}"
            cand_neighbors = set(graph.neighbors(cnode)) if cnode in graph else set()
            shared_neighbors = target_neighbors & cand_neighbors

            aa_raw = 0.0
            ra_raw = 0.0
            for z in shared_neighbors:
                w_uz = float(graph[target_node][z].get("weight", 0.25))
                w_vz = float(graph[cnode][z].get("weight", 0.25))
                deg_z = graph.degree(z)
                # Also incorporate actual group member count if z is a group node
                mc_z = graph.nodes[z].get("member_count")
                effective_pop = max(deg_z, int(mc_z) if isinstance(mc_z, int) else deg_z, 2)
                aa_raw += (w_uz * w_vz) / math.log(effective_pop + 1.0)
                ra_raw += (w_uz * w_vz) / max(deg_z, 1)

            if graph.has_edge(target_node, cnode):
                direct_w = float(graph[target_node][cnode].get("weight", 0.5))
                aa_raw += direct_w * 0.45
                ra_raw += direct_w * 0.35

            aa_norm = 1.0 - math.exp(-aa_raw / 0.55)
            ra_norm = 1.0 - math.exp(-ra_raw / 0.45)
            raw_ppr = ppr_scores.get(cnode, 0.0)
            ppr_norm = (raw_ppr / max_cand_ppr) if max_cand_ppr > 1e-9 else 0.0
            same_comm = 1.0 if (cnode in target_community and len(target_community) > 1) else 0.0

            metrics_by_uid[cand.user_id] = {
                "graph_adamic_adar": round(min(1.0, max(0.0, aa_norm)), 4),
                "graph_resource_allocation": round(min(1.0, max(0.0, ra_norm)), 4),
                "graph_personalized_pagerank": round(min(1.0, max(0.0, ppr_norm)), 4),
                "graph_same_community": round(same_comm, 4),
                "shared_graph_neighbors": float(len(shared_neighbors)),
            }

        summary = {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "user_nodes": len(all_accounts),
            "communities_detected": num_communities,
        }
        return metrics_by_uid, summary
