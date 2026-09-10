# `cpp/multimodal/` — vision/audio preprocessors and runtime runners

Multimodal runners are grouped **per model series**. Shared base classes and
stateless preprocessing utilities live in `common/`; everything else is owned by
the model series it belongs to.

| Directory | Model series | Runners |
|-----------|--------------|---------|
| `common/` | shared | `multimodalRunner` (base dispatch), `audioUtils`, `imageUtils`, `modelTypes` |
| `qwen2/` | Qwen2-VL | `qwenViTRunner` — also the base class the later Qwen VLs extend |
| `qwen2_5/` | Qwen2.5-VL | `qwen25vlViTRunner` |
| `qwen3/` | Qwen3-VL **and Qwen3.5** | `qwen3vlViTRunner` — one runner serves both; `ModelType::QWEN3_VL` and `ModelType::QWEN3_5` dispatch to `Qwen3VLViTRunner` |
| `qwen3_omni/` | Qwen3-Omni / Qwen3-TTS | `qwen3omniViTRunner`, `audioRunner`, `code2WavRunner`, `cloneEncoderRunner` |
| `gemma4/` | Gemma-4 | `gemma4ViTRunner`, `gemma4AudioRunner`, `gemma4UnifiedVisionRunner`, `gemma4UnifiedAudioRunner` |
| `nemotron_omni/` | Nemotron-Omni | `nemotronOmniViTRunner`, `nemotronOmniAudioRunner` |
| `phi4mm/` | Phi-4-multimodal | `phi4mmViTRunner` |
| `internvl/` | InternVL | `internViTRunner` |
| `cosmos3/` | Cosmos3-Edge | `cosmos3EdgeViTRunner` — reuses `Qwen3VLViTRunner` (from `qwen3/`) by inheritance |

Cross-series `#include`s are expected where one series extends another
(`qwen2_5`/`qwen3` include `qwen2/qwenViTRunner.h`; `qwen3_omni` and `cosmos3`
include `qwen3/qwen3vlViTRunner.h`). Runner selection lives in
`common/multimodalRunner.cpp`, keyed on `ModelType` (`common/modelTypes.h`).
