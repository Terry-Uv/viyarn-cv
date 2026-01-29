"""
This code was originally obtained from:
https://github.com/microsoft/Swin-Transformer
and
https://github.com/meta-llama/codellama/blob/main/llama/model.py

ViYaRN adaptation:
  (i)  band-wise frequency scaling (YaRN/NTK-by-parts style),
  (ii) axis-wise (x/y) calibration for anisotropic scaling,
  (iii) depth-wise phase scheduling (smoothly varying across blocks).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.distributed
from typing import Tuple

from timm.models.layers import to_2tuple

from .swin_transformer import SwinTransformer, SwinTransformerBlock, WindowAttention, BasicLayer
from .swin_transformer import PatchMerging

import torch.distributed as dist

_VIYARN_PRINTED_KEYS = set()

def rank0_print_once(msg: str, key: str | None = None):
    """
    Print only once on rank0 (or non-DDP).
    - key: 用来去重；不传则用 msg 本身做 key。
    """
    # DDP: only rank0
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return

    global _VIYARN_PRINTED_KEYS
    if key is None:
        key = msg
    if key in _VIYARN_PRINTED_KEYS:
        return
    _VIYARN_PRINTED_KEYS.add(key)
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def init_random_2d_freqs(head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    for _ in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)  # (2, nH, head_dim/2)
    return freqs


def compute_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor):
    # No float16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis  # (nH, N, head_dim/2) complex


def _smoothstep01(t: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _depth_ramp(alpha_max: float, block_idx: int, total_blocks: int, p: float = 1.0) -> float:
    """Monotone 0->alpha_max schedule across blocks (smoothstep^p)."""
    if total_blocks <= 1:
        return float(alpha_max)
    t = torch.tensor(block_idx / (total_blocks - 1), dtype=torch.float32)
    return float((_smoothstep01(t) ** p) * float(alpha_max))


def _blend_cis_by_phase(cis_base: torch.Tensor, cis_scaled: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Phase-smooth blending on the unit circle:
      cis = cis_base * exp(i * alpha * angle(cis_scaled / cis_base))
    """
    if alpha <= 0.0:
        return cis_base
    if alpha >= 1.0:
        return cis_scaled
    with torch.cuda.amp.autocast(enabled=False):
        delta = cis_scaled * torch.conj(cis_base)
        phase = torch.angle(delta)
        return cis_base * torch.polar(torch.ones_like(phase), phase * float(alpha))


def _viyarn_gamma(
    head_dim: int,
    *,
    yarn_gamma_lo: float,
    yarn_gamma_hi: float,
    yarn_transition: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (gamma, pow_term) per band.

    - head_dim is per-head *real* dim.
    - We treat each band as 4 real dims (x:2, y:2), so bands = head_dim//4.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for 2D axial RoPE, got {head_dim}.")
    bands = head_dim // 4
    idx = torch.arange(0, head_dim, 4, dtype=torch.float32, device=device)[:bands]  # 0,4,8,...
    if bands > 1:
        u = idx / idx[-1]  # u=0 high-freq, u=1 low-freq
    else:
        u = torch.zeros_like(idx)

    if yarn_transition == "cos":
        # u=0 -> gamma_hi, u=1 -> gamma_lo
        gamma = yarn_gamma_lo + (yarn_gamma_hi - yarn_gamma_lo) * 0.5 * (1.0 + torch.cos(torch.pi * u))
    elif yarn_transition == "linear":
        gamma = yarn_gamma_lo + (yarn_gamma_hi - yarn_gamma_lo) * u
    else:
        raise ValueError(f"Unsupported yarn_transition={yarn_transition!r}, use 'cos' or 'linear'.")

    pow_term = idx / float(head_dim)
    return gamma, pow_term


def _viyarn_band_scale(
    head_dim: int,
    *,
    end_x: int,
    end_y: int,
    base_end_x: int,
    base_end_y: int,
    yarn_gamma_lo: float,
    yarn_gamma_hi: float,
    yarn_transition: str,
    yarn_enable: bool,
    anisotropic: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (scale_x, scale_y) for applying on freqs[0] and freqs[1] respectively.

    When `yarn_enable`:
      theta_b = theta * s^gamma  =>  freq_scaled = freq_base * s^(-gamma * pow_term)
    So we return the multiplicative factor s^(-gamma * pow_term), expanded to head_dim/2 dims.
    """
    gamma, pow_term = _viyarn_gamma(
        head_dim, yarn_gamma_lo=yarn_gamma_lo, yarn_gamma_hi=yarn_gamma_hi, yarn_transition=yarn_transition, device=device
    )
    s_x = float(end_x) / float(base_end_x)
    s_y = float(end_y) / float(base_end_y)
    if not anisotropic:
        s = max(s_x, s_y)
        s_x = s_y = s

    if not yarn_enable:
        scale_band_x = torch.ones_like(gamma)
        scale_band_y = torch.ones_like(gamma)
    else:
        # per-band multiplicative factor on frequencies
        scale_band_x = (s_x ** (-gamma * pow_term)).to(dtype=torch.float32)
        scale_band_y = (s_y ** (-gamma * pow_term)).to(dtype=torch.float32)

    # both halves correspond to the same band set (see init_random_2d_freqs construction)
    scale_x = torch.cat([scale_band_x, scale_band_x], dim=-1)  # (head_dim/2,)
    scale_y = torch.cat([scale_band_y, scale_band_y], dim=-1)
    return scale_x, scale_y



# ---------------------------------------------------------------------------
# Mixed RoPE: freq-norm-aware NTK/YaRN scaling (for learnable mixed 2D freqs)
# ---------------------------------------------------------------------------

def _gamma_from_mixed_freqs_cos(
    freqs: torch.Tensor,           # (2, H, C)
    *,
    glo: float,
    ghi: float,
    rho_lo: float,
    rho_hi: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Return gamma per (head, channel) from the *magnitude* of learnable mixed freqs.

    - freqs: (2, H, C) where [0]=ωx, [1]=ωy
    - rho = log(||ω||); on rho-axis we cosine-smoothly interpolate gamma:
        z = clamp((rho - rho_lo) / (rho_hi - rho_lo), 0, 1)
        ramp = 0.5*(1+cos(pi*z))   # 1->0
        gamma = ghi + (glo-ghi)*ramp   (low-freq -> glo, high-freq -> ghi)
    """
    # monotone guard
    glo = float(glo); ghi = float(ghi)
    if glo < ghi:
        glo, ghi = ghi, glo

    wx, wy = freqs[0], freqs[1]                     # (H, C)
    omega = torch.sqrt(wx * wx + wy * wy) + eps     # (H, C)
    rho = torch.log(omega)

    denom = max(1e-6, float(rho_hi) - float(rho_lo))
    z = torch.clamp((rho - float(rho_lo)) / denom, 0.0, 1.0)
    ramp = 0.5 * (1.0 + torch.cos(torch.pi * z))
    return ghi + (glo - ghi) * ramp


def compute_mixed_cis_ntk_cos(
    freqs: torch.Tensor,           # (2, H, C)
    t_x: torch.Tensor, t_y: torch.Tensor,  # (N,)
    *,
    s_x: float,
    s_y: float,
    alpha: float,
    glo: float,
    ghi: float,
    rho_lo: float,
    rho_hi: float,
    anisotropic: bool = True,
) -> torch.Tensor:
    """
    Mixed 2D RoPE cis with NTK/YaRN-by-freq (non-PI) scaling:

      phase = (t_x * ωx) / s_x^{alpha*γ(‖ω‖)}  +  (t_y * ωy) / s_y^{alpha*γ(‖ω‖)}

    Returns:
      freqs_cis: (H, N, C) complex
    """
    # alpha guard
    a = float(alpha)
    if a <= 0.0:
        return compute_cis(freqs, t_x, t_y)

    device = t_x.device
    freqs = freqs.to(device=device)
    H, C = freqs.shape[1], freqs.shape[2]
    N = t_x.numel()

    gamma = _gamma_from_mixed_freqs_cos(freqs, glo=glo, ghi=ghi, rho_lo=rho_lo, rho_hi=rho_hi).to(device=device)
    exp_k = gamma * a                                 # (H, C)

    if anisotropic:
        sx = float(s_x); sy = float(s_y)
        scale_x = (sx ** exp_k).view(H, 1, C)
        scale_y = (sy ** exp_k).view(H, 1, C)
    else:
        s_edge = max(float(s_x), float(s_y))
        scale_x = scale_y = (s_edge ** exp_k).view(H, 1, C)

    tx = t_x.view(1, N, 1).to(device=device)
    ty = t_y.view(1, N, 1).to(device=device)
    wx = freqs[0].view(H, 1, C)
    wy = freqs[1].view(H, 1, C)

    phase = (wx * tx) / scale_x + (wy * ty) / scale_y  # (H, N, C)

    # avoid ComplexHalf
    with torch.cuda.amp.autocast(enabled=False):
        phase = phase.to(torch.float32)
        cis = torch.polar(torch.ones_like(phase), phase)
    return cis


def compute_axial_cis(
    head_dim: int,
    end_x: int,
    end_y: int,
    theta: float = 100.0,
    *,
    base_end_x: int | None = None,
    base_end_y: int | None = None,
    yarn_gamma_lo: float = 2.0,
    yarn_gamma_hi: float = 2.0,
    yarn_transition: str = "cos",
    yarn_enable: bool = True,
    anisotropic: bool = True,
) -> torch.Tensor:
    """
    Axial RoPE cis generator with NTK-by-parts (band-wise) scaling.

    Returns:
        freqs_cis: (N=end_x*end_y, head_dim/2) complex, shared across heads.
    """
    if base_end_x is None:
        base_end_x = end_x
    if base_end_y is None:
        base_end_y = end_y

    t_x, t_y = init_t_xy(end_x, end_y)
    device = t_x.device

    gamma, pow_term = _viyarn_gamma(
        head_dim, yarn_gamma_lo=yarn_gamma_lo, yarn_gamma_hi=yarn_gamma_hi, yarn_transition=yarn_transition, device=device
    )
    s_x = float(end_x) / float(base_end_x)
    s_y = float(end_y) / float(base_end_y)
    if not anisotropic:
        s = max(s_x, s_y)
        s_x = s_y = s

    if yarn_enable:
        theta_b_x = float(theta) * (s_x ** gamma)
        theta_b_y = float(theta) * (s_y ** gamma)
    else:
        theta_b_x = theta_b_y = torch.full_like(gamma, float(theta))

    freqs_x = 1.0 / (theta_b_x ** pow_term)
    freqs_y = 1.0 / (theta_b_y ** pow_term)

    with torch.cuda.amp.autocast(enabled=False):
        fx = torch.outer(t_x, freqs_x)  # (N, bands)
        fy = torch.outer(t_y, freqs_y)
        freqs_cis_x = torch.polar(torch.ones_like(fx), fx)
        freqs_cis_y = torch.polar(torch.ones_like(fy), fy)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    else:
        raise ValueError(f"Unexpected freqs_cis shape {tuple(freqs_cis.shape)} for x shape {tuple(x.shape)}")
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


# ---------------------------------------------------------------------------
# Swin RoPE (ViYaRN)
# ---------------------------------------------------------------------------

class RoPEWindowAttention(WindowAttention):
    def __init__(
        self,
        *args,
        rope_theta: float = 10.0,
        rope_mixed: bool = True,
        use_rpb: bool = False,
        # --- ViYaRN params ---
        viyarn_enable: bool = True,
        base_window_size: int | Tuple[int, int] | None = None,
        yarn_gamma_lo: float = 2.3,
        yarn_gamma_hi: float = 1.7,
        yarn_transition: str = "cos",
        anisotropic: bool = True,
        viyarn_alpha: float = 1.0,
        # --- Mixed NTK-by-freq params (for rope_mixed=True) ---
        mixed_ntk_enable: bool = True,
        mixed_rho_lo: float = -1.1643,
        mixed_rho_hi: float = 0.9026,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.rope_mixed = rope_mixed
        self.use_rpb = use_rpb
        if not self.use_rpb:
            self.relative_position_bias_table = None
            self.relative_position_index = None

        # window coords (local)
        t_x, t_y = init_t_xy(end_x=self.window_size[1], end_y=self.window_size[0])
        self.register_buffer("rope_t_x", t_x)
        self.register_buffer("rope_t_y", t_y)

        # ViYaRN config
        self.viyarn_enable = bool(viyarn_enable)
        self.viyarn_alpha = float(viyarn_alpha)
        self.yarn_gamma_lo = float(yarn_gamma_lo)
        self.yarn_gamma_hi = float(yarn_gamma_hi)
        self.yarn_transition = str(yarn_transition)
        self.anisotropic = bool(anisotropic)
        self.mixed_ntk_enable = bool(mixed_ntk_enable)
        # reuse yarn_gamma_{lo,hi} as mixed gamma endpoints by default
        self.mixed_glo = float(yarn_gamma_lo)
        self.mixed_ghi = float(yarn_gamma_hi)
        self.mixed_rho_lo = float(mixed_rho_lo)
        self.mixed_rho_hi = float(mixed_rho_hi)

        # window sizes: note WindowAttention.window_size is (Wh, Ww)
        end_y, end_x = int(self.window_size[0]), int(self.window_size[1])
        if base_window_size is None:
            base_y, base_x = end_y, end_x
        else:
            base_y, base_x = to_2tuple(base_window_size)
        self._viyarn_end_x = int(end_x)
        self._viyarn_end_y = int(end_y)
        self._viyarn_base_end_x = int(base_x)
        self._viyarn_base_end_y = int(base_y)
        # precompute scale ratios for mixed NTK-by-freq
        self._viyarn_sx = float(self._viyarn_end_x) / float(self._viyarn_base_end_x)
        self._viyarn_sy = float(self._viyarn_end_y) / float(self._viyarn_base_end_y)

        head_dim = self.dim // self.num_heads
        self._viyarn_head_dim = int(head_dim)

        # Rope freqs init (kept compatible with the original file).
        freqs = init_random_2d_freqs(head_dim=head_dim, num_heads=self.num_heads, theta=rope_theta, rotate=self.rope_mixed)
        if self.rope_mixed:
            self.rope_freqs = nn.Parameter(freqs, requires_grad=True)
        else:
            self.register_buffer("rope_freqs", freqs)

        need_scaling = (
            self.viyarn_enable
            and (self._viyarn_end_x != self._viyarn_base_end_x or self._viyarn_end_y != self._viyarn_base_end_y)
        )
        self._viyarn_need_scaling = bool(need_scaling)

        if self.rope_mixed:
            # For mixed RoPE, freqs are learnable; keep precomputed scale factors and blend in forward.
            if self._viyarn_need_scaling:
                scale_x, scale_y = _viyarn_band_scale(
                    head_dim,
                    end_x=self._viyarn_end_x,
                    end_y=self._viyarn_end_y,
                    base_end_x=self._viyarn_base_end_x,
                    base_end_y=self._viyarn_base_end_y,
                    yarn_gamma_lo=self.yarn_gamma_lo,
                    yarn_gamma_hi=self.yarn_gamma_hi,
                    yarn_transition=self.yarn_transition,
                    yarn_enable=True,
                    anisotropic=self.anisotropic,
                    device=self.rope_t_x.device,
                )
            else:
                scale_x = torch.ones(head_dim // 2, dtype=torch.float32, device=self.rope_t_x.device)
                scale_y = torch.ones(head_dim // 2, dtype=torch.float32, device=self.rope_t_x.device)

            # persistent=False: keep strict checkpoint loading compatible with the original RoPE-Swin.
            self.register_buffer("viyarn_scale_x", scale_x, persistent=False)
            self.register_buffer("viyarn_scale_y", scale_y, persistent=False)
        else:
            # For axial RoPE, precompute the per-block cis and keep it as a lightweight attribute (not in state_dict).
            cis_base = compute_axial_cis(
                head_dim,
                end_x=self._viyarn_end_x,
                end_y=self._viyarn_end_y,
                theta=rope_theta,
                base_end_x=self._viyarn_base_end_x,
                base_end_y=self._viyarn_base_end_y,
                yarn_enable=False,
                anisotropic=self.anisotropic,
                yarn_gamma_lo=self.yarn_gamma_lo,
                yarn_gamma_hi=self.yarn_gamma_hi,
                yarn_transition=self.yarn_transition,
            )
            if self._viyarn_need_scaling and self.viyarn_alpha > 0.0:
                cis_scaled = compute_axial_cis(
                    head_dim,
                    end_x=self._viyarn_end_x,
                    end_y=self._viyarn_end_y,
                    theta=rope_theta,
                    base_end_x=self._viyarn_base_end_x,
                    base_end_y=self._viyarn_base_end_y,
                    yarn_enable=True,
                    anisotropic=self.anisotropic,
                    yarn_gamma_lo=self.yarn_gamma_lo,
                    yarn_gamma_hi=self.yarn_gamma_hi,
                    yarn_transition=self.yarn_transition,
                )
                self.rope_freqs_cis = _blend_cis_by_phase(cis_base, cis_scaled, self.viyarn_alpha)
            else:
                self.rope_freqs_cis = cis_base

    def _scaled_freqs(self, freqs: torch.Tensor) -> torch.Tensor:
        # freqs: (2, nH, head_dim/2)
        if not self._viyarn_need_scaling:
            return freqs
        scale_x = self.viyarn_scale_x.to(device=freqs.device)
        scale_y = self.viyarn_scale_y.to(device=freqs.device)
        fx = freqs[0] * scale_x.unsqueeze(0)
        fy = freqs[1] * scale_y.unsqueeze(0)
        return torch.stack([fx, fy], dim=0)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)
            mask: (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        if self.rope_mixed:
            # Mixed RoPE (learnable freqs)
            if (not self._viyarn_need_scaling) or (self.viyarn_alpha <= 0.0) or (not self.viyarn_enable):
                freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
            else:
                # Two options:
                #   (A) freq-norm-aware NTK/YaRN (recommended for learnable mixed freqs)
                #   (B) legacy band-index scaling (kept for backward compatibility)
                if self.mixed_ntk_enable:
                    freqs_cis = compute_mixed_cis_ntk_cos(
                        self.rope_freqs,
                        self.rope_t_x,
                        self.rope_t_y,
                        s_x=self._viyarn_sx,
                        s_y=self._viyarn_sy,
                        alpha=self.viyarn_alpha,
                        glo=self.mixed_glo,
                        ghi=self.mixed_ghi,
                        rho_lo=self.mixed_rho_lo,
                        rho_hi=self.mixed_rho_hi,
                        anisotropic=self.anisotropic,
                    )
                else:
                    # Legacy: precomputed per-band multiplicative scaling + optional phase blending
                    if self.viyarn_alpha >= 1.0:
                        freqs_scaled = self._scaled_freqs(self.rope_freqs)
                        freqs_cis = compute_cis(freqs_scaled, self.rope_t_x, self.rope_t_y)
                    else:
                        cis_base = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
                        freqs_scaled = self._scaled_freqs(self.rope_freqs)
                        cis_scaled = compute_cis(freqs_scaled, self.rope_t_x, self.rope_t_y)
                        freqs_cis = _blend_cis_by_phase(cis_base, cis_scaled, self.viyarn_alpha)
        else:
            freqs_cis = self.rope_freqs_cis.to(x.device)

        q, k = apply_rotary_emb(q, k, freqs_cis)
        attn = q @ k.transpose(-2, -1)

        if self.use_rpb:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPESwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        fused_window_process=False,
        rope_theta=10.0,
        rope_mixed=True,
        use_rpb=False,
        # --- ViYaRN params ---
        viyarn_enable: bool = True,
        base_window_size: int | Tuple[int, int] | None = None,
        yarn_gamma_lo: float = 2.3,
        yarn_gamma_hi: float = 1.7,
        yarn_transition: str = "cos",
        anisotropic: bool = True,
        viyarn_alpha: float = 1.0,
        # --- Mixed NTK-by-freq params (for rope_mixed=True) ---
        mixed_ntk_enable: bool = True,
        mixed_rho_lo: float = -1.1643,
        mixed_rho_hi: float = 0.9026,
    ):
        super().__init__(
            dim,
            input_resolution,
            num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            fused_window_process=fused_window_process,
        )

        self.attn = RoPEWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            use_rpb=use_rpb,
            viyarn_enable=viyarn_enable,
            base_window_size=base_window_size,
            yarn_gamma_lo=yarn_gamma_lo,
            yarn_gamma_hi=yarn_gamma_hi,
            yarn_transition=yarn_transition,
            anisotropic=anisotropic,
            viyarn_alpha=viyarn_alpha,
            mixed_ntk_enable=mixed_ntk_enable,
            mixed_rho_lo=mixed_rho_lo,
            mixed_rho_hi=mixed_rho_hi,
        )


class RoPEBasicLayer(BasicLayer):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        fused_window_process=False,
        rope_theta=10.0,
        rope_mixed=True,
        use_rpb=False,
        # --- ViYaRN params ---
        viyarn_enable: bool = True,
        base_window_size: int | Tuple[int, int] | None = None,
        yarn_gamma_lo: float = 2.3,
        yarn_gamma_hi: float = 1.7,
        yarn_transition: str = "cos",
        anisotropic: bool = True,
        viyarn_alphas: Tuple[float, ...] | None = None,
        # --- Mixed NTK-by-freq params (for rope_mixed=True) ---
        mixed_ntk_enable: bool = True,
        mixed_rho_lo: float = -1.1643,
        mixed_rho_hi: float = 0.9026,
    ):
        super().__init__(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            fused_window_process=fused_window_process,
        )

        if viyarn_alphas is None:
            viyarn_alphas = tuple([1.0] * depth)
        if len(viyarn_alphas) != depth:
            raise ValueError(f"viyarn_alphas length {len(viyarn_alphas)} must equal depth {depth}.")

        # rebuild blocks with RoPE+ViYaRN attention
        self.blocks = nn.ModuleList(
            [
                RoPESwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    fused_window_process=fused_window_process,
                    rope_theta=rope_theta,
                    rope_mixed=rope_mixed,
                    use_rpb=use_rpb,
                    viyarn_enable=viyarn_enable,
                    base_window_size=base_window_size,
                    yarn_gamma_lo=yarn_gamma_lo,
                    yarn_gamma_hi=yarn_gamma_hi,
                    yarn_transition=yarn_transition,
                    anisotropic=anisotropic,
                    viyarn_alpha=float(viyarn_alphas[i]),
                    mixed_ntk_enable=mixed_ntk_enable,
                    mixed_rho_lo=mixed_rho_lo,
                    mixed_rho_hi=mixed_rho_hi,
                )
                for i in range(depth)
            ]
        )


class RoPESwinTransformer(SwinTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        fused_window_process=False,
        rope_theta=10.0,
        rope_mixed=True,
        use_rpb=False,
        # --- ViYaRN params ---
        viyarn_enable: bool = True,
        base_window_size: int | Tuple[int, int] | None = None,
        yarn_gamma_lo: float = 2.3,
        yarn_gamma_hi: float = 1.7,
        yarn_transition: str = "cos",
        anisotropic: bool = True,
        viyarn_depth_ramp_p: float = 1.0,
        viyarn_scale_threshold: float = 1.05,
        viyarn_alpha_max: float = 1.0,
        # --- Mixed NTK-by-freq params (for rope_mixed=True) ---
        mixed_ntk_enable: bool = True,
        mixed_rho_lo: float = -1.1643,
        mixed_rho_hi: float = 0.9026,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        # absolute position embedding
        self.ape = False
        self.absolute_pos_embed = None

        # stochastic depth
        total_blocks = int(sum(depths))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # depth-wise phase scheduling (global across the whole network)
        win = to_2tuple(window_size)
        end_y, end_x = int(win[0]), int(win[1])
        if base_window_size is None:
            base_y, base_x = end_y, end_x
        else:
            base_y, base_x = to_2tuple(base_window_size)

        s_x = float(end_x) / float(base_x)
        s_y = float(end_y) / float(base_y)
        s_edge = max(s_x, s_y)
        is_upsampling = bool(s_edge > float(viyarn_scale_threshold))
        alpha_max = float(viyarn_alpha_max) if (viyarn_enable and (s_edge != 1.0)) else 0.0
        if is_upsampling:
            viyarn_alphas_all = tuple(
                _depth_ramp(alpha_max, bi, total_blocks, p=float(viyarn_depth_ramp_p)) for bi in range(total_blocks)
            )
        else:
            viyarn_alphas_all = tuple(float(alpha_max) for _ in range(total_blocks))

        patches_resolution = self.patch_embed.patches_resolution

        rank0_print_once(
            (
                "[ViYaRN] init: "
                f"base_window_size={base_window_size}, window_size={window_size}, "
                f"s_edge={s_edge:.3f}, "
                f"gamma_lo={yarn_gamma_lo}, gamma_hi={yarn_gamma_hi}, transition={yarn_transition}, "
                f"depth_ramp_p={viyarn_depth_ramp_p}, scale_threshold={viyarn_scale_threshold}, "
                f"alpha_max={alpha_max}, "
                f"alpha_min={viyarn_alphas_all[0]:.3f}, alpha_max={viyarn_alphas_all[-1]:.3f}"  # 选择第一个和最后一个
            ),
            key="viyarn_init_once",   # 关键：所有 block 都共用同一个 key -> 全模型只打印一次
        )


        # build layers
        self.layers = nn.ModuleList()
        blk_offset = 0
        for i_layer in range(self.num_layers):
            depth = int(depths[i_layer])
            layer = RoPEBasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer), patches_resolution[1] // (2**i_layer)),
                depth=depth,
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[blk_offset : blk_offset + depth],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process,
                rope_theta=rope_theta,
                rope_mixed=rope_mixed,
                use_rpb=use_rpb,
                viyarn_enable=viyarn_enable,
                base_window_size=base_window_size,
                yarn_gamma_lo=yarn_gamma_lo,
                yarn_gamma_hi=yarn_gamma_hi,
                yarn_transition=yarn_transition,
                anisotropic=anisotropic,
                viyarn_alphas=viyarn_alphas_all[blk_offset : blk_offset + depth],
                mixed_ntk_enable=mixed_ntk_enable,
                mixed_rho_lo=mixed_rho_lo,
                mixed_rho_hi=mixed_rho_hi,
            )
            blk_offset += depth
            self.layers.append(layer)

        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rope_freqs", "relative_position_bias_table"}
