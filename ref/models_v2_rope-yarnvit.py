'''
发现glo和ghi反了，纠正过来的layerwise版本

small最佳：
| **rope_axial_small_ntk+glo=2.30+ghi=1.70** |      |      |      | **81.69%** | **81.84%** | **81.53%** | **79.73%** |

base最佳：
| **rope_axial_base_ntk+glo=1.90+ghi=1.70** |      |      |      | **84.21%** | **84.39%** | **84.03%** | **83.00%** |
| **rope_axial_base_ntk+glo=2.10+ghi=1.70** |      |      |      | **84.21%** | **84.39%** | **84.03%** | **83.00%** |
| **rope_axial_base_ntk+glo=2.30+ghi=1.70** |      |      |      | **84.21%** | **84.40%** | **84.04%** | **83.00%** |
| **rope_axial_base_ntk+glo=1.50+ghi=1.80** |      |      |      | **84.22%** | **84.39%** | **84.06%** | **83.00%** |

'''

"""
This code was originally obtained from:
https://github.com/facebookresearch/deit
and
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
p_hyper = 1

from functools import partial
from typing import Tuple

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from models_v2 import vit_models, Layer_scale_init_Block, Attention
from logger import log_once

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs, t_x, t_y, num_heads):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis


def compute_axial_cis(dim: int,
                      end_x: int,
                      end_y: int,
                      theta: float = 100.0,
                      *,
                      # --- NTK-by-parts additions ---
                      base_end_x: int = None,
                      base_end_y: int = None,
                      yarn_gamma_lo: float = 2,
                      yarn_gamma_hi: float = 2,
                      yarn_transition: str = "cos",
                      yarn_enable: bool = True,
                      anisotropic: bool = True):
    """
    Axial RoPE cis generator with NTK-by-parts (band-wise) scaling.

    Args:
        dim: per-head embedding dim.
        end_x/end_y: token-grid size along x/y at *current* resolution.
        theta: base RoPE theta used at base resolution.
        base_end_x/base_end_y: token-grid size at *base* resolution
            (e.g., 224/16=14). If None, defaults to current end_x/end_y.
        yarn_gamma_lo / yarn_gamma_hi: γ(u) 的低/高端点，控制低频放缩强度。
        yarn_transition: "cos"（推荐）或 "linear"；定义 γ(u) 在 u∈[0,1] 的过渡。
        yarn_enable: 置 False 等价于回退到原始 Axial RoPE。
        anisotropic: True 时按 s_x、s_y 各向异性放缩；否则用 s=max(s_x,s_y) 同步两轴。
    """
    log_once(f"========== Current Using Hyper: {yarn_enable} Yarn, low = {yarn_gamma_lo}, high = {yarn_gamma_hi}, anisotropic = {anisotropic}")
    # 1) base grid
    if base_end_x is None: base_end_x = end_x
    if base_end_y is None: base_end_y = end_y

    # 2) coordinates (no PI; pure RoPE)
    # 把 token 的 2D 坐标摊平成一维序列坐标 t_x, t_y。每个 token 都有一个 x 索引和 y 索引（0…end_x-1 / 0…end_y-1）。
    t_x, t_y = init_t_xy(end_x, end_y)

    # 3) frequency bands
    # 确定频带数 B，并用 idx 表示每个频带的索引。
    # 为什么 //4？——在2D轴向 RoPE 里，每个频带对应四个实数通道（x: cos/sin 两个，y: cos/sin 两个），
    # 所以每 4 维是一组频带。dim // 4 就是“这个头能承载的频带数”。
    B = dim // 4
    idx = torch.arange(0, dim, 4, dtype=torch.float32, device=t_x.device)[:B]  # 0,4,8,...
    # 把频带索引归一化到 [0,1] 的相对频率坐标 u，0 = 高频，1 = 低频。
    # 后面要让缩放强度随频率平滑变化（低频动得多，高频动得少），就需要这个标准化坐标。
    if B > 1:
        u = idx / idx[-1]   # 0..1, high->low freq
    else:
        u = torch.zeros_like(idx)

    
    # u = 0 高频处 gamma_high, u=1 低频处 gamma_low
    # 低频更依赖缩放（gamma_lo 大），高频少动（gamma_hi 小）。
    # 在 LLM 的 NTK-aware/YaRN 里，这样“低频多调、高频少调”的思想正是保证核近似不变、又避免高频震荡的关键。
    if yarn_transition == "cos":
        # ! wrong formula before: γ(u)=γ_hi + (γ_lo-γ_hi)*0.5*(1+cos(pi*u))
        gamma = yarn_gamma_lo + (yarn_gamma_hi - yarn_gamma_lo) * 0.5 * (1.0 + torch.cos(torch.pi * u))
    else:
        # 线性，1 - u，u越大越高频->1，那么gamma就减小->0，base不变
        gamma = yarn_gamma_lo + (yarn_gamma_hi - yarn_gamma_lo) * u

    # 4) resolution ratios
    # 各向异性算一下放大倍数s
    s_x = float(end_x) / float(base_end_x)
    s_y = float(end_y) / float(base_end_y)
    if anisotropic:
        s_x_eff, s_y_eff = s_x, s_y
    else:
        s = max(s_x, s_y)
        s_x_eff = s_y_eff = s

    # 5) per-band effective thetas
    # 每个频带，apply 不同转速调幅，调不同的base
    if yarn_enable:
        theta_b_x = theta * (s_x_eff ** gamma)
        theta_b_y = theta * (s_y_eff ** gamma)
    else:
        theta_b_x = theta_b_y = torch.full_like(gamma, float(theta))

    # 6) axial freqs per band
    # 把“有效base”映射成频带基频向量
    pow_term = (idx / float(dim))
    freqs_x = 1.0 / (theta_b_x ** pow_term)
    freqs_y = 1.0 / (theta_b_y ** pow_term)

    # 7) outer + complex phase
    with torch.cuda.amp.autocast(enabled=False):
        fx = torch.outer(t_x, freqs_x)
        fy = torch.outer(t_y, freqs_y)
        freqs_cis_x = torch.polar(torch.ones_like(fx), fx)
        freqs_cis_y = torch.polar(torch.ones_like(fy), fy)

    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

# --- ADD: helper functions for layer-wise smooth blending ---

def _smoothstep01(t: torch.Tensor):
    t = torch.clamp(t, 0, 1)
    return t * t * (3 - 2 * t)

def _depth_ramp(alpha_max: float, layer_idx: int, L: int, p: float = 2.0) -> float:
    """Single scalar for this layer: 0->alpha_max, monotone & smooth."""
    if L <= 1: 
        return float(alpha_max)
    t = torch.tensor(layer_idx / (L - 1), dtype=torch.float32)
    return float(_smoothstep01(t) ** p * alpha_max)

# def _alpha_from_scale(s_edge: float) -> float:
#     """
#     Target max strength given resolution scale s (>=1 means upsampling).
#     Use piecewise-smooth ramps to avoid jumps.
#     1.05 ~ 256, 1.30 ~ 320, 1.70 ~ 384, >= ~512.
#     """
#     a0, a1, a2 = 1.05, 1.30, 1.70
#     if s_edge <= a0:
#         return 0.0
#     elif s_edge <= a1:
#         # ramp to ~0.6
#         z = (s_edge - a0) / (a1 - a0)
#         return float(0.6 * _smoothstep01(torch.tensor(z)))
#     elif s_edge <= a2:
#         # ramp further to ~0.85
#         z = (s_edge - a1) / (a2 - a1)
#         return float(0.6 + 0.25 * _smoothstep01(torch.tensor(z)))
#     else:
#         return 1.0  # strongest at >=512
def _alpha_from_scale(s: float) -> float:
    # 对称门：下采样也给一点点 α，最多 ~0.15
    # if s < 1.0:
    #     z = (1.0 - s) / (1.0 - 0.64)  # 0.64≈144/224
    #     # return float(0.2+0.4 * _smoothstep01(torch.tensor(z)))
    #     return 1.2

    # # 上采样分段：256/320/384/512 逐级拔高
    # a0, a1, a2, a3 = 1.05, 1.30, 1.70, 2.286  # ~256, ~320, ~384, ~512
    # if s <= a0:
    #     return 0.0
    # elif s <= a1:
    #     z = (s - a0) / (a1 - a0)
    #     return float(0.20 + 0.40 * _smoothstep01(torch.tensor(z)))  # → ~0.6
    # elif s <= a2:
    #     z = (s - a1) / (a2 - a1)
    #     return float(0.60 + 0.25 * _smoothstep01(torch.tensor(z)))  # → ~0.85
    # elif s <= a3:
    #     z = (s - a2) / (a3 - a2)
    #     return float(0.85 + 0.15 * _smoothstep01(torch.tensor(z)))  # → ~1.0
    # else:
    #     return 1.0
    return 1.0


def _blend_cis_by_phase(cis_base: torch.Tensor, cis_scaled: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Phase-smooth blending on the unit circle:
      cis = cis_base * exp( i * alpha * angle(cis_scaled / cis_base) )
    Keeps |cis|=1 and avoids head-wise mixing.
    """
    with torch.cuda.amp.autocast(enabled=False):
        delta = cis_scaled * torch.conj(cis_base)
        phase = torch.angle(delta)              # real tensor
        return cis_base * torch.polar(torch.ones_like(phase), phase * float(alpha))



def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

# 原始的、正确的函数
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class RoPEAttention(Attention):
    """Multi-head Attention block with relative position embeddings."""
    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = RoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        
        return x

class rope_vit_models(vit_models):
    def __init__(self, rope_theta=100.0, rope_mixed=False, use_ape=False,
                 base_img_size = 224,
                 yarn_gamma_lo: float = 2,
                 yarn_gamma_hi: float = 2,
                 yarn_transition: str = 'cos',
                 anisotropic: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        img_size = kwargs['img_size'] if 'img_size' in kwargs else 224
        patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else 16
        num_heads = kwargs['num_heads'] if 'num_heads' in kwargs else 12
        embed_dim = kwargs['embed_dim'] if 'embed_dim' in kwargs else 768
        mlp_ratio = kwargs['mlp_ratio'] if 'mlp_ratio' in kwargs else 4.
        
        base_end_x = base_img_size // patch_size
        base_end_y = base_img_size // patch_size
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        
        self.use_ape = use_ape
        if not self.use_ape:
            self.pos_embed = None            
        
        self.rope_mixed = rope_mixed
        self.num_heads = num_heads
        self.patch_size = patch_size
        
        self.base_end_x = base_end_x
        self.base_end_y = base_end_y
        self.anisotropic = anisotropic
        self.yarn_gamma_lo = yarn_gamma_lo
        self.yarn_gamma_hi = yarn_gamma_hi
        self.yarn_transition = yarn_transition
        self.rope_theta = rope_theta
        
        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
            
            freqs = []
            for i, _ in enumerate(self.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            t_x, t_y = init_t_xy(end_x = img_size // patch_size, end_y = img_size // patch_size)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            # >>> changed: pass base_end_x/base_end_y so NTK-by-parts knows the training grid <<<
            self.compute_cis = partial(
                compute_axial_cis,
                dim=embed_dim//num_heads,
                theta=rope_theta,
                base_end_x=base_end_x,
                base_end_y=base_end_y,
                # yarn_gamma_lo=yarn_gamma_lo,
                anisotropic=anisotropic
            )
            
            freqs_cis = self.compute_cis(end_x = img_size // patch_size, end_y = img_size // patch_size)
            self.freqs_cis = freqs_cis

    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'freqs'}
        
    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_ape:
            pos_embed = self.pos_embed
            if pos_embed.shape[-2] != x.shape[-2]:
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                pos_embed = pos_embed.view(
                    1, (img_size[1] // patch_size[1]), (img_size[0] // patch_size[0]), self.embed_dim
                ).permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed, size=(H // patch_size[1], W // patch_size[0]), mode='bicubic', align_corners=False
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            x = x + pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.rope_mixed:
            if self.freqs_t_x.shape[0] != x.shape[1] - 1:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i])
        
        # else:
        #     if self.freqs_cis.shape[0] != x.shape[1] - 1:
        #         freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size)
        #     else:
        #         freqs_cis = self.freqs_cis
        #     freqs_cis = freqs_cis.to(x.device)
            
        #     for i , blk in enumerate(self.blocks):
        #         x = blk(x, freqs_cis=freqs_cis)
                
        else:
            # --- Layer-wise, resolution-aware phase blending (all heads unified) ---
            gx, gy = W // self.patch_size, H // self.patch_size

            # base (no NTK scaling) & scaled (with NTK scaling) at current resolution
            cis_base   = self.compute_cis(end_x=gx, end_y=gy,
                                          yarn_enable=False).to(x.device)
            cis_scaled = self.compute_cis(end_x=gx, end_y=gy,
                                          yarn_enable=True,
                                          yarn_gamma_lo=self.yarn_gamma_lo,
                                          yarn_gamma_hi=self.yarn_gamma_hi,
                                          yarn_transition=self.yarn_transition).to(x.device)

            # resolution factor for alpha_max
            s_x = float(gx) / float(self.base_end_x)
            s_y = float(gy) / float(self.base_end_y)
            s_edge = max(s_x, s_y) if self.anisotropic else max(s_x, s_y)  # 同步/各向一致时也用max更保守
            alpha_max = _alpha_from_scale(s_edge)
            is_upsampling = s_edge > 1.05

            L = len(self.blocks)
            for li, blk in enumerate(self.blocks):
                if is_upsampling:
                    # 高分辨率：也就是原本的 Ours，保持平滑过渡
                    alpha_li = _depth_ramp(alpha_max, li, L, p=p_hyper)
                else:
                    # 低分辨率/原分辨率：也就是 Ablation 2，直接一步到位
                    alpha_li = alpha_max
                freqs_cis_li = _blend_cis_by_phase(cis_base, cis_scaled, alpha_li)
                x = blk(x, freqs_cis=freqs_cis_li)

                
        x = self.norm(x)
        x = x[:, 0]
        
        return x

# RoPE-Axial
@register_model
def rope_axial_small_ntk(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    # 让命令行 / 外部传进来的同名参数优先生效，避免 multiple values 错误
    base_img_size = kwargs.pop('base_img_size', 224)
    yarn_gamma_lo = kwargs.pop('yarn_gamma_lo', 2.3)
    yarn_gamma_hi = kwargs.pop('yarn_gamma_hi', 1.7)
    
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False,
        base_img_size = base_img_size,
        yarn_gamma_lo = yarn_gamma_lo,
        yarn_gamma_hi = yarn_gamma_hi,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def rope_axial_base_ntk(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    base_img_size = kwargs.pop('base_img_size', 224)
    yarn_gamma_lo = kwargs.pop('yarn_gamma_lo', 2.3)
    yarn_gamma_hi = kwargs.pop('yarn_gamma_hi', 1.7)
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False, 
        base_img_size = base_img_size,
        yarn_gamma_lo = yarn_gamma_lo,
        yarn_gamma_hi = yarn_gamma_hi,
        **kwargs)
    return model

@register_model
def rope_axial_large_ntk(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    base_img_size = kwargs.pop('base_img_size', 224)
    yarn_gamma_lo = kwargs.pop('yarn_gamma_lo', 2.3)
    yarn_gamma_hi = kwargs.pop('yarn_gamma_hi', 1.7)
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False,
        base_img_size = base_img_size,
        yarn_gamma_lo = yarn_gamma_lo,
        yarn_gamma_hi = yarn_gamma_hi,
        **kwargs)
    return model

@register_model
def rope_axial_deit_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False, **kwargs)
    return model

@register_model
def rope_axial_deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False, **kwargs)
    return model


@register_model
def rope_axial_deit_small_patch16_LS_im100(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False, **kwargs)
    model.default_cfg = _cfg()
    return model



if __name__ == "__main__":
    from timm.models import create_model

    # 实例化模型
    model = create_model('rope_axial_small_ntk', num_classes=1000)
    print(model)

    # 前向传播测试
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(output.shape)  # 应输出 torch.Size([2, 1000])

    # 参数量对比
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    # 原始DeiT-Small: ~22M，DCMHA版本: ~23.4M