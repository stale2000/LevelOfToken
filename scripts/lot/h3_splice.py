"""Target-video tail for a Fizgig ``MiniMaxH3DiT``.

This module does not import Fizgig. The DiT loads it only when ``_lot`` is set.

Latent space. The adapter works in y-space: each extent's clean patch is the
pretrained latent divided by that extent's scale ``s`` (eq. 16). Until a
Procrustes fit every ``s`` is 1 and y-space is the pretrained latent. After
``fit_extent`` it is not. Then the sampler state handed to the DiT must be
``y_t``, built from ``to_y(x0)``, and the clean estimate goes back through
``to_x``. A splice built with ``y_space=False`` refuses fitted scales instead
of silently feeding x-space rows to y-space heads.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from adapter import LotVisualAdapter
from h3 import patchify, unpatchify
from h3_positions import video_positions
from layout import LotLayout


def h3_clean_estimate(y_t: torch.Tensor, x0_minus_eps: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Recover H3's clean estimate from its ``x0 - eps`` output in y-space."""
    if y_t.shape != x0_minus_eps.shape:
        raise ValueError("H3 state and output must have the same shape")
    sig = torch.as_tensor(sigma, device=y_t.device, dtype=y_t.dtype)
    if sig.ndim == 1 and sig.shape[0] == y_t.shape[0]:
        sig = sig.reshape(y_t.shape[0], *([1] * (y_t.ndim - 1)))
    try:
        sig = torch.broadcast_to(sig, y_t.shape)
    except RuntimeError as exc:
        raise ValueError("sigma is not broadcastable to the H3 state") from exc
    if not torch.isfinite(sig).all() or bool((sig < 0).any()) or bool((sig > 1).any()):
        raise ValueError("sigma must be finite and in [0, 1]")
    return y_t + sig * x0_minus_eps


def transition_layout_state(
    y_t: torch.Tensor,
    clean_x0: torch.Tensor,
    sigma: torch.Tensor | float,
    old_splice: LotSplice,
    new_splice: LotSplice,
) -> torch.Tensor:
    """Change LoT layout coordinates while preserving the estimated noise.

    This converts only the clean component of the Euler state because LoT's
    fitted extent scales apply to clean latents, not to the sampled noise.
    ``clean_x0`` is expressed in pretrained H3 latent coordinates.
    """
    if y_t.shape != clean_x0.shape:
        raise ValueError("H3 state and clean estimate must have the same shape")
    if old_splice.layout == new_splice.layout:
        return y_t
    sig = torch.as_tensor(sigma, device=y_t.device, dtype=y_t.dtype)
    if sig.ndim == 1 and sig.shape[0] == y_t.shape[0]:
        sig = sig.reshape(y_t.shape[0], *([1] * (y_t.ndim - 1)))
    try:
        sig = torch.broadcast_to(sig, y_t.shape)
    except RuntimeError as exc:
        raise ValueError("sigma is not broadcastable to the H3 state") from exc
    if not torch.isfinite(sig).all() or bool((sig < 0).any()) or bool((sig > 1).any()):
        raise ValueError("sigma must be finite and in [0, 1]")
    old_clean = old_splice.to_y(clean_x0).to(device=y_t.device, dtype=y_t.dtype)
    new_clean = new_splice.to_y(clean_x0).to(device=y_t.device, dtype=y_t.dtype)
    return y_t + (1.0 - sig) * (new_clean - old_clean)


class LotSplice:
    """One layout and the adapter that embeds and scatters its video tail."""

    def __init__(self, adapter: LotVisualAdapter, layout: LotLayout, *, y_space: bool = False):
        self.adapter = adapter
        self.layout = layout
        self.y_space = bool(y_space)

    def unit_scales(self) -> bool:
        """True while every extent in this layout still has ``s = 1``."""
        used = {rect.extent for rect in self.layout.rects}
        return all(float(self.adapter.bank.scale(extent)) == 1.0 for extent in used)

    def to_y(self, x0: torch.Tensor) -> torch.Tensor:
        """Pretrained clean latent ``(B, C, T, H, W)`` to y-space, same shape."""
        tokens = patchify(x0)
        return unpatchify(self.adapter.bank.scale_clean(tokens, self.layout)).to(x0.dtype)

    def to_x(self, y0: torch.Tensor) -> torch.Tensor:
        """Inverse of ``to_y``: a y-space clean estimate back to the pretrained latent."""
        tokens = patchify(y0)
        return unpatchify(self.adapter.bank.unscale(tokens, self.layout)).to(y0.dtype)

    def video_tokens(self, video_rows: torch.Tensor, latent_t: int, lat_h: int, lat_w: int) -> torch.Tensor:
        """``(n, 96)`` patch rows from ``patchify_video`` to ``(1, T, H, W, 96)``."""
        if not self.y_space and not self.unit_scales():
            raise ValueError(
                "extent scales are fitted (s != 1), so the DiT input must be y_t. "
                "Build the state with to_y and pass LotSplice(..., y_space=True)."
            )
        token_h, token_w = lat_h // 2, lat_w // 2
        if lat_h % 2 or lat_w % 2:
            raise ValueError("latent sides must be divisible by the 2x2 patch")
        expected = latent_t * token_h * token_w
        if video_rows.ndim != 2 or video_rows.shape[0] != expected:
            raise ValueError(f"video rows {tuple(video_rows.shape)} != ({expected}, D)")
        if video_rows.shape[-1] != self.adapter.token_dim:
            raise ValueError(f"token dim {video_rows.shape[-1]} != {self.adapter.token_dim}")
        if self.layout.time != latent_t or self.layout.height != token_h or self.layout.width != token_w:
            raise ValueError(
                f"layout {(self.layout.time, self.layout.height, self.layout.width)} "
                f"does not match tokens {(latent_t, token_h, token_w)}"
            )
        return video_rows.reshape(1, latent_t, token_h, token_w, video_rows.shape[-1])

    def embed_rows(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(L, hidden)`` for the packed sequence. The batch dimension is 1."""
        hidden, _centers = self.adapter.embed(tokens, self.layout)
        return hidden[0]

    def replace_video_positions(
        self,
        dense_pos: torch.Tensor,
        latent_t: int,
        lat_h: int,
        lat_w: int,
    ) -> torch.Tensor:
        """Keep the text, condition, and audio prefix. Replace the dense video tail."""
        n_dense = latent_t * (lat_h // 2) * (lat_w // 2)
        if dense_pos.shape[0] < n_dense:
            raise ValueError("position tensor is shorter than the dense video tail")
        origin = float(dense_pos[-n_dense, 0].item())
        tail = video_positions(self.layout, lat_h, lat_w, origin)
        return torch.cat([dense_pos[:-n_dense], tail.to(dtype=dense_pos.dtype)], dim=0)

    def modulate(self, final_layer, states: torch.Tensor, t_emb: torch.Tensor, t_index: int) -> torch.Tensor:
        """RMSNorm and AdaLN, the shared part of ``FinalLayer``. Not ``video_out``."""
        shift, scale = final_layer.adaln_proj(t_emb)
        return final_layer.norm(states) * (1.0 + scale[t_index]) + shift[t_index]

    def project_rows(self, modulated: torch.Tensor, tokens: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        """Extent linears plus the dense velocity, flattened in patchify order.

        The result is H3's head convention, ``x0 - eps``, so the Fizgig sampler
        steps it unchanged. It is in the same space as ``tokens``: y-space once
        scales are fitted.
        """
        weight = next(iter(self.adapter.out_proj.parameters()))
        states = modulated.to(dtype=weight.dtype, device=weight.device).unsqueeze(0)
        tokens = tokens.to(dtype=weight.dtype, device=weight.device)
        velocity = self.adapter.velocity_from_states(
            states, tokens, sigma, self.layout, x0_minus_eps=True)
        return velocity.reshape(-1, velocity.shape[-1])


class BudgetedLayoutController:
    """Select a new LoT layout after an Euler update, using its clean estimate.

    The host sampler calls ``after_euler_step`` at a step boundary and installs
    the returned splice before the next DiT forward. A zero refresh interval
    selects once after warm-up and then freezes the layout.
    """

    def __init__(
        self,
        splice: LotSplice,
        budget_per_frame: int,
        *,
        policy: str = "entropy",
        calibration=None,
        warmup_steps: int = 5,
        warmup_sigma: float | None = None,
        refresh_interval: int = 0,
        bins: int = 16,
        seed: int = 0,
        selection_latent: torch.Tensor | None = None,
        shuffle_roots: bool = False,
    ) -> None:
        if warmup_steps < 0 or refresh_interval < 0:
            raise ValueError("warmup and refresh steps must be non-negative")
        if warmup_sigma is not None and (not math.isfinite(warmup_sigma) or not 0 <= warmup_sigma <= 1):
            raise ValueError("warmup_sigma must be finite and in [0, 1]")
        if not isinstance(budget_per_frame, int):
            raise TypeError("budget_per_frame must be an integer")
        self.splice = splice
        self.budget_per_frame = int(budget_per_frame)
        self.policy = policy
        self.calibration = calibration
        self.warmup_steps = int(warmup_steps)
        self.warmup_sigma = warmup_sigma
        self.refresh_interval = int(refresh_interval)
        self.bins = int(bins)
        self.seed = int(seed)
        self.selection_latent = None if selection_latent is None else selection_latent.detach().cpu()
        self.shuffle_roots = bool(shuffle_roots)
        self.last_selection_step: int | None = None
        self.last_selection = None

    @torch.no_grad()
    def after_euler_step(
        self,
        state_before: torch.Tensor,
        output: torch.Tensor,
        sigma_before: torch.Tensor | float,
        state_after: torch.Tensor,
        sigma_after: torch.Tensor | float,
        completed_steps: int,
    ):
        """Return `(state, active_splice, selection)` for the next Euler step."""
        if completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        if state_before.shape != state_after.shape:
            raise ValueError("Euler states before and after must have matching shapes")
        crossed_sigma = self.warmup_sigma is not None and float(torch.as_tensor(sigma_after)) <= self.warmup_sigma
        warmup_complete = crossed_sigma if self.warmup_sigma is not None else completed_steps >= self.warmup_steps
        due = warmup_complete and (
            self.last_selection_step is None
            or self.refresh_interval > 0
            and completed_steps - self.last_selection_step >= self.refresh_interval
        )
        if not due:
            return state_after, self.splice, None

        from budget_layout import choose_budget_layout, permute_root_layout

        clean_y = h3_clean_estimate(state_before, output, sigma_before)
        clean_x = self.splice.to_x(clean_y)
        selection_x = clean_x if self.selection_latent is None else self.selection_latent
        if selection_x.shape != clean_x.shape:
            raise ValueError("selection latent must match the sampled video latent shape")
        token_lattice = patchify(selection_x)[0]
        selection = choose_budget_layout(
            token_lattice,
            self.budget_per_frame,
            policy=self.policy,
            calibration=self.calibration,
            bins=self.bins,
            seed=self.seed + completed_steps,
        )
        if self.shuffle_roots:
            selection = replace(
                selection,
                layout=permute_root_layout(selection.layout, seed=self.seed + completed_steps),
                policy=selection.policy + "_root_shuffled",
            )
        self.last_selection_step = completed_steps
        self.last_selection = selection
        if selection.layout == self.splice.layout:
            return state_after, self.splice, selection

        next_splice = LotSplice(self.splice.adapter, selection.layout, y_space=True)
        changed = transition_layout_state(
            state_after, clean_x, sigma_after, self.splice, next_splice
        )
        self.splice = next_splice
        return changed, next_splice, selection
