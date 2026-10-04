# Dense H3 transfer candidate

H3 assigns one area-normalized spatial coordinate to each 2x2 transformer patch.
The learned upscaler's Conv3d encoder produces a dense feature field. Treating
its even/odd cells as four independently resampled lanes creates separated
copies of narrow features. At 36->50, one source row at index 18 becomes two
equal peaks at target rows 24 and 26 with a zero between them. Dense coordinate
ramps at 36x54->50x76 alternate between 1.0 and 0.4305 source-cell increments.
A continuous coordinate transform must have uniform increments.

The selected path now uses `h3_dense_patch_center_lattice_v2`. For a dense axis
of length n and spatial area A, its cell coordinates are
`16*(1-n/sqrt(A)) + (i-0.5)*32/sqrt(A)`. The mean of each adjacent cell pair is
the native H3 patch coordinate. Bilinear sampling acts on the complete dense
field between encoder and decoder. Flow uses the identical transform for its
source prefix carrier. No fitted transform or final output warp is applied.

The provider requires `h3_patch_lattice_api=2`; general clean-video API version 1
and the callable `upscale_clean_video_h3_patch_lattice` remain. Flow rejects
the earlier phase-separated capability before sampling, so both companion
overlays must be updated. Ordinary upscaler node/provider calls retain their
trained half-pixel interpolation. Parameter keys, normalization and temporal
network execution are unchanged. Interpolation scratch is bounded to roughly
64 MiB per chunk; the network processes the complete temporal sequence once.

Regression evidence covers contiguous unimodal edge support in both axes and
both spatial phases, uniform dense coordinates, native patch centers, both
orientations, dtype/batch/time ownership, decoder-input routing, and repeated
geometry changes through the same cached network. The v1 implementation fails
all 14 edge/spacing cases even though all four coordinate round-trip cases pass.
Round-trip equality alone cannot qualify either individual operator.

Runs 01138 and 01139 both execute v1 with zero user-LoRA hooks. They differ in
seed, prompt, first reference and orientation; they do not isolate a repeat-run
state defect. The new tests reproduce an operator defect without trained
weights. The attached metrics contain no rendered frames or tensor bytes, so
the cause of every visible artifact in those runs remains unproven. Trained
checkpoint continuity, texture, tone, time and peak VRAM require runtime
validation. Replay the failing landscape workflow with both updated overlays
and verify the v2 encoder-to-decoder log, then repeat its unchanged settings.
