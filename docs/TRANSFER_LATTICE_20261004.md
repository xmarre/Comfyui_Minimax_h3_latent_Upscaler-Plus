# Physical H3 transfer candidate

Flow's partitioned continuation creates its source prefix on MiniMax H3's
area-normalized, endpoint-excluded 2x2 patch lattice. The learned 3D upscaler
previously always interpolated encoded features with trilinear half-pixel
coordinates. These maps are different, including phase and axis scale when
rounded source and target aspect ratios differ. Restoring the authoritative
target prefix therefore joins differently sampled coordinate domains.

The provider now advertises `h3_patch_lattice_api=1` and exposes
`upscale_clean_video_h3_patch_lattice`. This selects physical patch transport
between encoder and decoder convolutions. All temporal blocks still process
the complete sequence once. The four within-patch features retain their
ownership. Border extension matches Flow's prefix construction. Interpolation
scratch is split into bounded batches; this does not chunk the learned network.
Ordinary node/provider calls keep their original half-pixel behavior.

Coordinate-field round trips at 46x40 -> 66x58 and 54x36 -> 76x50 fail with the
old mixed conventions and pass with physical transport in both directions.
Tests also compare the decoder's actual input with each selected transport,
exercise fp32/fp16/bf16 and batch/temporal/patch ownership, and verify one
provider invocation. The checkpoint architecture and parameter keys do not
change.

This repairs a demonstrated coordinate contract; it does not qualify a trained
checkpoint's rendered continuity or texture/tone response. Decoder weights were
trained with the ordinary interpolation path. A matched real workflow must
verify the new path, including time and peak VRAM. No GPU cost is measured here.
The zero-user-LoRA run 01132 is the primary replay; a subsequent matched
user-LoRA on/off pair remains necessary.

The actual selected path emits `[H3 learned transfer]` with
`spatial_lattice=h3_physical_patch_lattice_v1`, source/target H/W, temporal
length, and `resample_position=encoder_to_decoder`. Flow separately counts the
provider call and measures same-frame learned-prefix/authoritative-prefix affine
correspondence before the overwrite. Neither receipt certifies visual success.
