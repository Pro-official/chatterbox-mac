"""
MLX-PyTorch Hybrid S3Gen.

S3Gen is complex (Conformer + U-Net + Flow Matching + HiFiGAN).
For practical performance gains, we use a hybrid approach:
- MLX: VoiceEncoder + T3 (biggest bottleneck)
- PyTorch/MPS: S3Gen + HiFiGAN (already efficient on MPS)

This gives ~80% of the performance benefit with much less complexity.
"""

# S3Gen stays in PyTorch for now - it runs efficiently on MPS
# The integration is handled in the main ChatterboxMLX class
