
import sys
import os
import torch

# Add ref folder to sys.path so we can import swin_transformer_rope_viyarn1
ref_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ref"))
if ref_path not in sys.path:
    sys.path.append(ref_path)

try:
    from swin_transformer_rope_viyarn1 import RoPESwinTransformer
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_model():
    print("Instantiating model...")
    # Instantiate with small model for speed
    model = RoPESwinTransformer(
        img_size=224, 
        embed_dim=96, 
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24],
        window_size=7,
        rope_mixed=True,
        viyarn_enable=True,
        base_window_size=7,
        # Force scaling to trigger logic
        viyarn_scale_threshold=1.05 
    )
    model.eval()
    
    # 1. Test standard size
    x = torch.randn(1, 3, 224, 224)
    print(f"Testing forward pass with input shape {x.shape}...")
    try:
        out = model(x)
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Test larger size (upsampling)
    # 448x448 -> patch grid 112x112. patch_size=4. 
    # Swin uses window_size=7. 
    # This should work fine and trigger any resolution-dependent scaling if implemented in forward
    x_large = torch.randn(1, 3, 448, 448)
    print(f"Testing forward pass with input shape {x_large.shape} (upsampling)...")
    try:
        out_large = model(x_large)
        print(f"Output shape: {out_large.shape}")
    except Exception as e:
        print(f"Large forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Verification passed!")

if __name__ == "__main__":
    test_model()
