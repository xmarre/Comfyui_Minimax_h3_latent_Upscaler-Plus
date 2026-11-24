<p align="center">
  <a href="./README.md">English</a> ·
  <a href="./README_zh.md"><strong>中文</strong></a>
</p>

# ComfyUI MiniMax H3 Latent Upscaler-Plus

用于 ComfyUI 的 MiniMax H3 视频 latent 学习式空间放大，并可选集成低 sigma 的 H3 精修采样。

> **Plus 分支：** 这是 `xmarre` 维护的 [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) 分支。它使用相同的学习网络和权重，并增加了 H3 感知的精修、H3 Continuum 互通、采样器内部交接 provider 以及若干正确性修复。部分行为有意与上游不同，见[与上游的差异](#与上游的差异)。

训练好的网络直接放大 24 通道 H3 视频 latent。两阶段工作流因此可以先在较低分辨率生成，在 latent 空间放大，再在目标分辨率做简短精修，无需 VAE 解码 → 像素放大 → VAE 编码的往返。音频从不经过放大网络。

学习式放大节省的是时间。它不会降低后续 H3 采样所需的显存，后续采样仍在完整的目标分辨率网格上运行 transformer。

**示例：** [视频对比（MP4）](examples/Minimax_h3_latent_Upscaler_001.mp4)

![图像放大对比](examples/Minimax_h3_latent_Upscaler_002.jpg)

## 目录

- [节点](#节点)
- [安装](#安装)
- [工作流](#工作流)
- [行为说明](#行为说明)
- [节点参考](#节点参考)
- [Provider API](#provider-api)
- [模型与训练数据](#模型与训练数据)
- [与上游的差异](#与上游的差异)
- [测试](#测试)
- [H3 协同发布集](#h3-协同发布集)
- [致谢](#致谢)

## 节点

所有节点都位于 `video/MinimaxH3` 分类下。

| 显示名称 | 节点 ID | 用途 |
| --- | --- | --- |
| Minimax H3 Latent Upscaler (2D) | `MinimaxH3LatentUpscalerNode2D` | 2D 主干加时间层的学习式放大。仅支持倍率模式。 |
| Minimax H3 Latent Upscaler (3D) | `MinimaxH3LatentUpscaler3D` | 完整 3D 学习式放大。支持倍率、目标尺寸或百万像素模式，并做双轴对齐。 |
| MiniMax H3 Latent Upscaler + Refine (3D) | `MinimaxH3LatentUpscaler3DRefineHandoff` | 学习式 3D 放大后执行 H3 精修采样，输出可直接解码的 LATENT。支持原生联合 AV latent 和 H3 Continuum 分段列表。 |
| MiniMax H3 Latent Upscaler Provider (3D) [Experimental] | `MinimaxH3LatentUpscaler3DProvider` | 采样器内部学习式交接的配置对象，例如 Flow-Aligned Regenerate 的 `learned_3d` 迁移。自身不采样。 |

2D 和 3D 节点是普通的 `LATENT → LATENT` 放大器，不运行 H3。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus.git Comfyui_Minimax_h3_latent_Upscaler
```

安装后重启 ComfyUI。标准 ComfyUI 之外唯一的依赖是 `einops`，ComfyUI 已自带。

### 权重

把权重放到 `ComfyUI/models/latent_upscale_models/`，或在 ComfyUI 的 `extra_model_paths.yaml` 中通过 `latent_upscale_models` 键配置其他目录。2D、3D 放大节点及集成细化/交接节点均使用 ComfyUI 的递归模型发现机制查找 `.safetensors` 和 `.pth` 文件。子目录中的权重保留相对路径。如果多个目录存在相同的相对文件名，则按 ComfyUI 注册的路径顺序加载；标准目录尚未注册时会作为后备目录添加。更改模型路径或替换已缓存的权重后，请重启 ComfyUI。

预训练权重：[huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

| 文件 | 存储精度 |
| --- | --- |
| `minimax_h3_latent_upscaler_3d_bf16.safetensors` | bf16 |
| `minimax_h3_latent_upscaler_3d_fp16.safetensors` | fp16 |
| `minimax_h3_latent_upscaler_3d_fp32.pth` | fp32 |

这些文件是同一个 3D 结构，供 3D、Refine 和 Provider 节点使用。若存在 bf16 文件，Provider 会默认选中它。`precision` 控件设置的是推理精度，与文件的存储精度相互独立。

加载器会从权重中检测块数、通道宽度和时间层。2D 节点需要 2D 主干结构的权重；在 2D 节点中加载 3D 权重会报缺少权重的错误。

## 工作流

### 单独放大

2D 和 3D 节点接收纯 24 通道视频 latent（`B×24×T×H×W`，单帧为 `B×24×H×W`）。联合 H3 音视频 latent 需先拆分，放大后再合并：

```text
联合 H3 AV latent
  → LTXVSeparateAVLatent
      video_latent → Minimax H3 Latent Upscaler (3D) ─┐
      audio_latent ───────────────────────────────────┤
  → LTXVConcatAVLatent
  → 自选的第二次 H3 采样，或 VAE Decode
```

[`workflow_templates/`](workflow_templates/) 中的图生视频和参考生视频工作流使用这一方式，之后用 `BasicGuider` 和 `SamplerCustomAdvanced` 在外部完成第二次采样。

### 在一个节点中放大并精修

**MiniMax H3 Latent Upscaler + Refine (3D)** 取代了“拆分 / 放大 / 合并 / 第二个采样器”这一串节点：

```text
低分辨率 H3 latent ─► MiniMax H3 Latent Upscaler + Refine (3D) ─► VAE Decode
                        + noise    (RandomNoise)
                        + sampler  (KSamplerSelect)
                        + sigmas   (部分去噪调度，sigmas[0] < 1)
                        + model + positive   （原生工作流）
                          或 refine_state    （H3 Continuum）
```

原生工作流连接联合 AV 的 `latent`、`model` 和 `positive`，`audio_latent` 保持断开。`negative` 可选：不连接时节点只用正向条件引导采样，这是 H3 的常规用法；连接后节点按 `cfg` 使用 CFG。

不要再外接 `BasicGuider`、`DisableNoise` 或 `SamplerCustomAdvanced`，节点内部自己完成采样。

### H3 Continuum

连接 [H3 Continuum-Plus](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus) 中 **H3 Continuum Sampler V3.4** 的并行分段列表输出：

```text
H3 Continuum Sampler V3.4
  video_latents ──► Refine.latent
  audio_latents ──► Refine.audio_latent
  refine_state  ──► Refine.refine_state
```

连接 `refine_state` 后，Continuum 会为每个分段捕获一个新的 MODEL 克隆（带有该分段的 Continuum 选项）以及第一次采样实际使用的正向条件。Continuum 还会把每个分段的视频/音频去噪掩码附加到 latent 输出上。已连接的 `refine_state` 优先：残留的 `model`、`positive` 或 `negative` 连接会被忽略；`refine_state` 格式错误时直接报错，而不会退回使用这些连接。

节点按顺序处理整个分段列表。如果某个分段以完全受保护的续接前缀开头（整段时间步的去噪掩码为 0），在第二次采样之前，该前缀会被替换为上一分段*精修后*输出的最后几个时间步。这样相邻分段衔接的是精修后的内容，而不是各自独立放大的副本。`lock_audio` 关闭时，音频前缀也按同样方式承接。承接要求各分段的视频几何尺寸一致。

Run Storage 不保存精修状态。连接 `refine_state` 时如果 Continuum 复用已存储的分段，会直接报错。请关闭 Run Storage，或从第 1 段重新生成。

### 采样器内部交接（实验性）

**MiniMax H3 Latent Upscaler Provider (3D)** 输出一个 `H3_LATENT_UPSCALER` 对象。把它连接到接受该类型的消费者，例如 [MiniMax-H3 Flow-Aligned Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate) 的 Target Input 渐进交接，并在消费者中选择学习式迁移模式（`learned_3d`）。消费者在每次交接时对其干净视频估计调用一次 provider，并指定精确的目标 latent 尺寸。Provider 从不接收音频，也不运行任何 H3 步。推荐的交接设置见消费者的文档。Flow-Aligned Regenerate 的 Partitioned Exact-Prefix Handoff 在每个 Continuum 续接块中也会调用它：其默认的 `progressive_uniform_source` 模式（Flow v0.3.11 起）在每个续接块通过 provider 迁移整个生成轨迹一次，因此请保持 provider 连接。

## 行为说明

### 输出尺寸与对齐

3D 和 Refine 节点根据所选模式计算像素空间目标：

- `scale by multiplier`：源尺寸 × `scale`。
- `target dimensions`：`width` × `height`。
- `megapixels`：按源宽高比取 `megapixels × 1024²` 个像素。

`align` 是像素空间要求。H3 的 VAE 还需要 16 像素网格，因此两个轴都放在 `lcm(align, 16)` 网格上。默认 `align=32` 对应 32 像素网格；`align=24` 对应 48 像素网格。

- `keep_proportion=False`：宽和高分别取整到网格。
- `keep_proportion=True`：在附近的网格组合中搜索，选出最能保持源宽高比且接近目标的一组，两个轴都在网格上。

只支持放大。任一轴目标小于源尺寸都会被拒绝。结果与源尺寸相同时，原样返回 latent。

2D 节点按 `scale` 缩放 latent 网格并取整，不做像素网格对齐。

### 精修采样

Refine 节点依次：

1. 用学习式 3D 网络只放大视频分量；
2. 可选地把学习模型移到 CPU（`offload_after_upscale`）；
3. 在放大后的网格上重建联合 H3 AV latent；
4. 在放大后的网格上重建去噪掩码（见[音频与掩码](#音频与掩码)）；
5. 调整目标网格条件的尺寸（见[条件几何](#条件几何)）；
6. 用 `noise` 为放大后的 AV 网格生成新噪声；
7. 通过 ComfyUI 标准 guider 路径，用给定的 `sampler` 按 `sigmas` 采样。噪声缩放由 ComfyUI 按模型自身的方式完成，因此没有手动预加噪，也没有 `DisableNoise` 阶段。

`sigmas[0]` 必须满足 `0 <= sigmas[0] < 1`。在 H3 的 flow 参数化下，从 1.0 开始的调度会让放大后的 latent 权重为零，因此会被拒绝；请使用部分去噪调度。`SIGMAS` 为空时，直接返回放大后的 latent，不采样。

第二次采样在 MODEL 的克隆上运行，该克隆的 `transformer_options["h3_refinement"]` 把这次调用标记为精修（API 1，`sigma_reference` 为模型的 `sigma_max`）。配套的运行时补丁会读取这一约定，因此把第二次采样当作精修而不是一次全新生成。原 MODEL 不会被修改。

第二次采样的开销随目标 token 数增长。空间放大 2× 时，每个 H3 步的视频 token 约为原来的 4 倍。请保持精修调度简短，并在自己的硬件上测试。

### 音频与掩码

`lock_audio`（默认开启）：

- **开启**：音频噪声为零，音频去噪掩码为零，采样后精确恢复第一次采样的音频。
- **关闭**：音频获得正常噪声并与视频一起精修，保留已有的音频去噪掩码。

已有的视频去噪掩码会用最近邻方式缩放到放大后的网格，因此完全受保护的区域保持受保护，Native-Masked 续接前缀不会被重新去噪。没有掩码且 `lock_audio` 关闭时，节点不带掩码采样。

### 条件几何

- `minimax_keyframes` 是目标网格条件，会被调整到 H3 内部填充后的偶数 latent 网格（H3 会把 H/W 填充到其 2×2 patch 网格）。
- `minimax_refs` 是独立的参考块，有自己的 latent 尺寸和 RoPE 网格，保持不变。
- 条件会被复制，不会被原地修改；容器类型和附加字段保持不变。

放大后的 latent 本身保持学习输出的精确形状，H3 在内部会把填充部分裁回。

### 模型缓存、设备与卸载

- 加载后的网络按 `(权重, 设备, 精度)` 缓存，并在多次运行之间复用。
- `offload_after_upscale`（3D、Refine 和 Provider；默认关闭）在使用后把缓存的网络移到 CPU，下次使用时再移回。在 Refine 节点上，卸载发生在放大之后、第二次采样之前，此处回收显存最有意义。显存足够时请保持关闭，否则每次运行都要付出传输开销。
- 没有 CUDA 时，2D、3D 和 Refine 节点会退回 CPU。Provider 不会静默切换设备，而是直接报错。
- 3D 网络始终一次处理完整的时间序列，从不切分为时间分块。
- 推理时禁用学习网络中的注意力层。

## 节点参考

### Minimax H3 Latent Upscaler (2D)

| 输入 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `latent` | LATENT | | 24 通道视频 latent |
| `model_name` | 下拉 | | 2D 主干权重 |
| `scale` | FLOAT | 2.0 | 1.0–4.0 |
| `device` | 下拉 | `cuda` | `cuda`、`cpu` |
| `precision` | 下拉 | `fp32` | `fp32`、`fp16`、`bf16` |

输出：`LATENT`。

### Minimax H3 Latent Upscaler (3D)

| 输入 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `latent` | LATENT | | 24 通道视频 latent，5D 或 4D |
| `model_name` | 下拉 | | 3D 权重 |
| `mode` | 下拉 | `scale by multiplier` | 显示 `scale`（1.0–4.0）、`width`/`height`（64–4096）或 `megapixels`（0.1–8.0） |
| `align` | INT | 32 | 像素对齐，按 `lcm(align, 16)` 组合 |
| `keep_proportion` | BOOLEAN | true | |
| `device` | 下拉 | `cuda` | `cuda`、`cpu` |
| `precision` | 下拉 | `fp16` | `fp32`、`fp16`、`bf16` |
| `offload_after_upscale` | BOOLEAN | false | |

尺寸输入的默认值：`scale` 2.0，`width` 1280，`height` 704，`megapixels` 1.0。输出：`LATENT`。

### MiniMax H3 Latent Upscaler + Refine (3D)

包含 3D 节点的 `model_name`、`mode`、`scale`、`width`、`height`、`megapixels`、`align`、`keep_proportion`、`device`、`precision`（默认 `fp16`）和 `offload_after_upscale`。所有尺寸输入都会显示，但只有所选 `mode` 用到的才生效。其他输入：

| 输入 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `latent` | LATENT | 是 | 原生联合 AV latent，或拆分输入时的视频 latent |
| `noise` | NOISE | 是 | 例如 `RandomNoise` |
| `sampler` | SAMPLER | 是 | 例如 `KSamplerSelect` |
| `sigmas` | SIGMAS | 是 | 部分去噪调度，`sigmas[0] < 1` |
| `lock_audio` | BOOLEAN | 是 | 默认 true |
| `cfg` | FLOAT | 是 | 默认 1.0；仅在原生路径连接了 `negative` 时使用 |
| `audio_latent` | LATENT | 仅拆分输入 | `latent` 只含视频时必需；联合 AV latent 时必须断开 |
| `refine_state` | H3_CONTINUUM_REFINE_STATE | Continuum | 优先于 `model`/`positive`/`negative` |
| `model` | MODEL | 原生 | 无 `refine_state` 时必需 |
| `positive` | CONDITIONING | 原生 | 无 `refine_state` 时必需 |
| `negative` | CONDITIONING | 否 | 仅原生路径；启用 CFG |

输出：`LATENT` 列表，为最终可直接解码的结果。单个 latent 输入得到只含一项的列表。

### MiniMax H3 Latent Upscaler Provider (3D) [Experimental]

| 输入 | 类型 | 默认值 |
| --- | --- | --- |
| `model_name` | 下拉 | 存在时为 `minimax_h3_latent_upscaler_3d_bf16.safetensors` |
| `device` | 下拉 | `cuda` |
| `precision` | 下拉 | `bf16` |
| `offload_after_upscale` | BOOLEAN | false |

输出：`H3_LATENT_UPSCALER`。

## Provider API

`H3_LATENT_UPSCALER` 的值是一个不可变的 `H3LatentUpscalerProvider`（`nodes/minimax_h3_handoff_provider.py`）。它只保存配置；权重加载和模型缓存留在本包内部。

| 属性 | 值 |
| --- | --- |
| `kind` | `"minimax_h3_learned_latent_upscaler"` |
| `api_version` | `1` |
| `h3_patch_lattice_api` | `2` |
| `model_name`、`device`、`precision`、`offload_after_upscale` | 节点输入 |

- `upscale_clean_video(video, *, target_h, target_w)`：接收干净的浮点 `B×24×T×H×W` 视频 latent，精确输出 `target_h × target_w`，保持 batch、通道、时间和 dtype。任一轴缩小会被拒绝；结果含非有限值时报错。
- `upscale_clean_video_h3_patch_lattice(video, *, target_h, target_w)`：运行同一网络，但在解码器之前，把稠密编码器特征重采样到这样一组坐标上：相邻单元坐标的均值等于 H3 原生 patch 中心（`h3_dense_patch_center_lattice_v2`）。要求空间轴为偶数，供需要在交接处匹配 H3 patch 坐标映射的消费者使用。普通节点和 provider 路径保持训练时的半像素插值。背景说明：[`docs/TRANSFER_LATTICE_20261004.md`](docs/TRANSFER_LATTICE_20261004.md)。

## 模型与训练数据

- **输入：** 24 通道 H3 视频 latent；推理前用训练时的逐通道统计量归一化，推理后反归一化。
- **3D 结构（默认配置；加载器从权重读取实际值）：** `in_channels=24`，编码器和解码器各 12 个 512 通道残差块，每隔一个残差块后接一个时间卷积（kernel 5），以及学习的倍率嵌入。
- **重采样：** 3D 网络在编码器和解码器之间做三线性插值，2D 网络做双线性插值。时间轴保持不变，只缩放 H×W。

上游权重在约 80,000 对样本（低分辨率 latent 与高分辨率目标）上训练：

| 类型 | 对数 | 占比 |
| --- | --- | --- |
| 视频片段 | ~70,000 | ~87.5% |
| 2K 图像 | ~8,000 | ~10% |

| 倍率 | 占比 |
| --- | --- |
| 2× | 40% |
| 1.5×、2.5×、3×、4× | 各 10% |
| 任意 1.0×–4.0× | 10% |

## 与上游的差异

上游的改动会经过审查后选择性采纳。目前有意保留的差异：

- **不做时间分块。** 3D 网络对完整序列只运行一次。其堆叠的 3D/时间卷积和 GroupNorm 统计量跨越时间维度，带部分重叠的分块执行与完整序列执行并不等价，还可能在分块边界产生差异。
- **卸载需手动开启。** `offload_after_upscale` 默认关闭，而不是每次运行后都卸载，以免重复运行或大显存环境反复传输。
- **双轴对齐保留 `keep_proportion`。** 两个轴都对齐到 `lcm(align, 16)`，同时保留宽高比锁定。
- **归一化在私有副本上原地进行。** 相同 dtype 下改为非原地运算只会增加整幅临时张量，不会提高数值精度。
- **H3 专属新增**（上游没有）：Refine 节点、Continuum 互通和交接 Provider。

## 测试

GitHub Actions 在推送到 `main` 和每个 pull request 时运行：在多个固定的 ComfyUI 源码版本上使用 Python 3.12，并在其中一个版本上使用 Python 3.10、3.11 和 3.13。每个任务运行 Ruff 和 `compileall`、原生 ComfyUI 源码契约测试以及回归测试套件。

测试覆盖：原生与拆分 AV 校验、学习式放大的委托调用、精确输出几何与对齐、关键帧与参考条件、去噪掩码重建、Continuum `refine_state` 解析与优先级、顺序分段前缀承接、guider 与采样器调用、部分去噪保护、锁定音频恢复、完整序列执行、缓存设备恢复、卸载时机、provider 契约以及 H3 patch 坐标变换。

测试不加载训练好的权重。输出质量、速度和峰值显存取决于工作负载和硬件，需要在真实工作流中检验。

## H3 协同发布集

本包与其他 H3 组件一起发布。各版本详情见 [RELEASE_NOTES.md](RELEASE_NOTES.md)。

| 组件 | 版本 | 包含的 PR |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.11](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.11) | [#97](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/97) |
| Sol-H3 | [v0.1.10](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.10) | [#40](https://github.com/xmarre/ComfyUI-Sol-H3/pull/40) |
| VDN-H3-Plus | [v1.5.9](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.9) | [#39](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/39), [#40](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/40) |
| H3 Continuum-Plus | [v3.4.6](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.6) | 未变更 |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | 未变更 |

[Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28) 是未变更的配套组件。独立的 Keyless、音频训练以及已否决的解码几何实验不在本发布集中。

经过测试的 Core 适配器修复是 [ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783)。使用 INT8 融合 MLP 运行时适配器时，请保留该 ComfyUI Patcher PR overlay，直到上游包含此修复。独立的 Core #16720 优化不在本发布集中。

## 致谢

- [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)：原始节点、网络和预训练权重。
- [Ttl/ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale)：神经网络 latent 放大的思路。
- LTX 2.3 Spatial Upscaler（`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`）：网络结构参考。
- 精修集成参考了 [Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler](https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler) 和 ComfyUI 的 MiniMax H3 采样代码，但为独立实现。本包不依赖 Tr1dae 或 Mamad8 的放大包。
