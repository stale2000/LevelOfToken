"""Content-guided LoT layouts under an exact per-frame token budget.

Input is an unscaled H3 token lattice ``(T, H, W, D)``. The selector builds
the same supported spatial quadtree for every scoring policy, then chooses the
highest-scoring valid refinement at the largest feasible count under budget.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from layout import LotLayout, TokenRect, layout_from_rects


@dataclass(frozen=True)
class EntropyCalibration:
    """Per-feature statistics fitted on training latents and reused at inference."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, samples: Iterable[torch.Tensor], eps: float = 1e-6) -> EntropyCalibration:
        total = total_sq = None
        count = 0
        for sample in samples:
            values = _check_tokens(sample).double().reshape(-1, sample.shape[-1]).cpu()
            if total is None:
                total = torch.zeros(values.shape[-1], dtype=torch.float64)
                total_sq = torch.zeros_like(total)
            if values.shape[-1] != total.numel():
                raise ValueError("all calibration samples must have the same feature width")
            total += values.sum(dim=0)
            total_sq += values.square().sum(dim=0)
            count += values.shape[0]
        if not count or total is None or total_sq is None:
            raise ValueError("entropy calibration needs at least one token sample")
        mean = total / count
        variance = (total_sq / count - mean.square()).clamp_min(float(eps) ** 2)
        return cls(mean.float(), variance.sqrt().float())


@dataclass(frozen=True)
class BudgetLayout:
    layout: LotLayout
    policy: str
    requested_per_frame: int
    actual_per_frame: int
    mean_refinement_score: float


@dataclass
class _Node:
    t: int
    u: int
    v: int
    h: int
    w: int
    children: list[_Node] = field(default_factory=list)
    score: float = 0.0
    # count -> (best utility, child token counts); None child counts means leaf.
    options: dict[int, tuple[float, tuple[int, ...] | None]] = field(default_factory=dict)


def _check_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 4 or not tokens.is_floating_point():
        raise ValueError("tokens must be a floating point (T, H, W, D) tensor")
    if min(tokens.shape) < 1:
        raise ValueError("token lattice axes must be positive")
    if not torch.isfinite(tokens).all():
        raise ValueError("token lattice contains non-finite values")
    return tokens.detach().float()


def _axis_roots(length: int) -> list[tuple[int, int]]:
    """Greedy supported chunks `(start, extent)` with a 4/2/1 boundary rule."""
    chunks = []
    start = 0
    for extent in (4, 2, 1):
        while length - start >= extent:
            chunks.append((start, extent))
            start += extent
    return chunks


def _build_node(t: int, u: int, v: int, h: int, w: int) -> _Node:
    node = _Node(t, u, v, h, w)
    child_h, child_w = max(1, h // 2), max(1, w // 2)
    if (child_h, child_w) == (h, w):
        return node
    for du in range(0, h, child_h):
        for dv in range(0, w, child_w):
            node.children.append(_build_node(t, u + du, v + dv, child_h, child_w))
    return node


def _root_nodes(time: int, height: int, width: int) -> list[list[_Node]]:
    rows, cols = _axis_roots(height), _axis_roots(width)
    return [
        [_build_node(t, u, v, h, w) for u, h in rows for v, w in cols]
        for t in range(time)
    ]


def _possible_counts(node: _Node) -> set[int]:
    if not node.children:
        return {1}
    counts = {0}
    for child in node.children:
        counts = {left + right for left in counts for right in _possible_counts(child)}
    return counts | {1}


def _feasible_frame_counts(roots: list[_Node]) -> set[int]:
    counts = {0}
    for root in roots:
        counts = {left + right for left in counts for right in _possible_counts(root)}
    return counts


def _entropy(values: torch.Tensor, calibration: EntropyCalibration, bins: int) -> float:
    """Mean per-feature discrete entropy; a spatially flat feature has zero entropy."""
    features = values.reshape(-1, values.shape[-1])
    mean = calibration.mean.to(dtype=features.dtype)
    std = calibration.std.to(dtype=features.dtype)
    if mean.numel() != features.shape[-1] or std.numel() != features.shape[-1]:
        raise ValueError("entropy calibration feature width does not match tokens")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("entropy calibration must have finite means and positive standard deviations")
    normalized = ((features - mean) / std).clamp(-3, 3)
    indices = (((normalized + 3) * (bins / 6)).floor().long()).clamp_(0, bins - 1)
    channels = torch.arange(features.shape[-1]).view(1, -1)
    codes = (channels * bins + indices).reshape(-1)
    counts = torch.bincount(codes, minlength=features.shape[-1] * bins)
    hist = counts.reshape(features.shape[-1], bins).float()
    probs = hist / hist.sum(dim=-1, keepdim=True).clamp_min(1)
    per_feature = -(probs * probs.clamp_min(1e-12).log2()).sum(dim=-1)
    return float(per_feature.mean())


def _node_score(
    node: _Node,
    tokens: torch.Tensor,
    policy: str,
    calibration: EntropyCalibration | None,
    random_field: torch.Tensor | None,
    bins: int,
) -> float:
    block = tokens[node.t, node.u:node.u + node.h, node.v:node.v + node.w]
    if policy == "entropy":
        if calibration is None:
            raise ValueError("entropy policy requires calibration fitted on training latents")
        return _entropy(block, calibration, bins)
    if policy == "detail":
        centered = block - block.mean(dim=(0, 1), keepdim=True)
        return float(centered.square().mean().sqrt())
    if policy == "random":
        assert random_field is not None
        return float(random_field[node.t, node.u:node.u + node.h, node.v:node.v + node.w].mean())
    raise ValueError(f"unknown policy {policy!r}; expected entropy, detail, or random")


def _score_tree(
    node: _Node,
    tokens: torch.Tensor,
    policy: str,
    calibration: EntropyCalibration | None,
    random_field: torch.Tensor | None,
    bins: int,
) -> None:
    node.score = _node_score(node, tokens, policy, calibration, random_field, bins)
    for child in node.children:
        _score_tree(child, tokens, policy, calibration, random_field, bins)


def _node_options(node: _Node) -> dict[int, tuple[float, tuple[int, ...] | None]]:
    options: dict[int, tuple[float, tuple[int, ...] | None]] = {1: (0.0, None)}
    if not node.children:
        node.options = options
        return options

    combined: dict[int, tuple[float, tuple[tuple[int, ...], ...]]] = {0: (0.0, ())}
    for child in node.children:
        child_options = _node_options(child)
        next_combined: dict[int, tuple[float, tuple[tuple[int, ...], ...]]] = {}
        for used, (utility, paths) in combined.items():
            for child_count, (child_utility, _child_path) in child_options.items():
                total = used + child_count
                value = utility + child_utility
                old = next_combined.get(total)
                if old is None or value > old[0] + 1e-12:
                    next_combined[total] = (value, paths + ((child_count,),))
        combined = next_combined

    split_reward = node.score * (node.h * node.w)
    for count, (utility, paths) in combined.items():
        child_counts = tuple(path[0] for path in paths)
        value = utility + split_reward
        old = options.get(count)
        if old is None or value > old[0] + 1e-12:
            options[count] = (value, child_counts)
    node.options = options
    return options


def _select_root_counts(roots: list[_Node], target: int) -> list[int]:
    # Stage-local backpointers avoid copying an ever-growing plan at every state.
    tables: list[dict[int, tuple[float, int, int]]] = [{0: (0.0, -1, -1)}]
    for root in roots:
        current: dict[int, tuple[float, int, int]] = {}
        for used, (utility, _prev, _choice) in tables[-1].items():
            for choice, (node_utility, _children) in root.options.items():
                total = used + choice
                if total > target:
                    continue
                value = utility + node_utility
                old = current.get(total)
                if old is None or value > old[0] + 1e-12:
                    current[total] = (value, used, choice)
        tables.append(current)
    if target not in tables[-1]:
        raise RuntimeError(f"layout budget {target} is not reachable from the selected roots")
    choices = []
    count = target
    for stage in range(len(roots), 0, -1):
        _utility, previous, choice = tables[stage][count]
        choices.append(choice)
        count = previous
    return list(reversed(choices))


def _emit(node: _Node, count: int, output: list[TokenRect]) -> float:
    utility, child_counts = node.options[count]
    if child_counts is None:
        output.append(TokenRect(node.t, node.u, node.v, 1, node.h, node.w))
        return utility
    for child, child_count in zip(node.children, child_counts):
        _emit(child, child_count, output)
    return utility


def choose_budget_layout(
    tokens: torch.Tensor,
    budget_per_frame: int,
    *,
    policy: str = "entropy",
    calibration: EntropyCalibration | None = None,
    bins: int = 16,
    seed: int = 0,
) -> BudgetLayout:
    """Select a supported rectangular quadtree under a strict per-frame cap.

    ``tokens`` is unscaled ``(T, H, W, D)`` H3 latent data. The target count is
    chosen from geometry alone before any content score is evaluated, ensuring
    different policies produce exactly budget-matched layouts.
    """
    values = _check_tokens(tokens).cpu()
    if policy not in {"entropy", "detail", "random"}:
        raise ValueError("policy must be entropy, detail, or random")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if not isinstance(budget_per_frame, int):
        raise TypeError("budget_per_frame must be an integer")
    time, height, width, _features = values.shape
    roots_by_frame = _root_nodes(time, height, width)
    feasible = _feasible_frame_counts(roots_by_frame[0])
    minimum, maximum = min(feasible), max(feasible)
    if budget_per_frame < minimum or budget_per_frame > maximum:
        raise ValueError(f"budget {budget_per_frame} outside feasible range [{minimum}, {maximum}]")
    target = max(count for count in feasible if count <= budget_per_frame)

    random_field = None
    if policy == "random":
        random_field = torch.rand(
            (time, height, width), generator=torch.Generator().manual_seed(seed)
        )

    utility_total = 0.0
    rects: list[TokenRect] = []
    for roots in roots_by_frame:
        for root in roots:
            _score_tree(root, values, policy, calibration, random_field, bins)
            _node_options(root)
        root_counts = _select_root_counts(roots, target)
        for root, count in zip(roots, root_counts):
            utility_total += _emit(root, count, rects)

    layout = layout_from_rects(time, height, width, rects)
    if layout.count != target * time:
        raise RuntimeError(f"constructed {layout.count} tokens, expected {target * time}")
    return BudgetLayout(
        layout=layout,
        policy=policy,
        requested_per_frame=budget_per_frame,
        actual_per_frame=target,
        mean_refinement_score=utility_total / time,
    )


def permute_root_layout(layout: LotLayout, seed: int, cell: int = 4) -> LotLayout:
    """Shuffle equally shaped quadtree roots with one spatial map across time.

    Reusing the same root permutation for every latent frame preserves temporal
    correspondence. Independent per-frame permutations create an artificial
    layout jitter control that can cause video flicker and ghosting.
    """
    import random
    from collections import defaultdict

    if cell < 1:
        raise ValueError("cell size must be positive")
    roots_by_shape = defaultdict(list)
    token_root = {}
    for u0 in range(0, layout.height, cell):
        for v0 in range(0, layout.width, cell):
            h, w = min(cell, layout.height - u0), min(cell, layout.width - v0)
            roots_by_shape[(h, w)].append((u0, v0))
            for t in range(layout.time):
                for u in range(u0, u0 + h):
                    for v in range(v0, v0 + w):
                        token_root[(t, u, v)] = (u0, v0, h, w)
    grouped = defaultdict(list)
    for rect in layout.rects:
        root = token_root.get((rect.t, rect.u, rect.v))
        if root is None:
            raise ValueError("layout rectangle falls outside a quadtree root")
        grouped[(rect.t, *root)].append(rect)
    rng = random.Random(seed)
    destination_by_shape = {}
    for shape, roots in sorted(roots_by_shape.items()):
        destinations = list(roots)
        rng.shuffle(destinations)
        destination_by_shape[shape] = dict(zip(roots, destinations))
    transformed = []
    for (t, src_u, src_v, h, w), rects in sorted(grouped.items()):
        dst_u, dst_v = destination_by_shape[(h, w)][(src_u, src_v)]
        for rect in rects:
            transformed.append(TokenRect(
                rect.t, dst_u + rect.u - src_u, dst_v + rect.v - src_v,
                rect.et, rect.eh, rect.ew,
            ))
    result = layout_from_rects(layout.time, layout.height, layout.width, transformed)
    if result.count != layout.count:
        raise RuntimeError("root permutation changed the token count")
    return result
