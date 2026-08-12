# Calibration Dataset Customization

Quantization calibration data is selected **by name**, per modality. The
quantization CLI passes only a dataset name:

```bash
tensorrt-edgellm-quantize llm \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --quantization nvfp4 \
  --text_dataset cnn_dailymail        # text
  # --image_dataset mmmu              # when --visual_quantization is set
  # --audio_dataset librispeech       # for Qwen3-ASR / audio quantization
```

A name maps to a registered dataset. If the name is not registered,
quantization **fails out** with the list of available datasets and a pointer
back to this guide and the template — it does not silently fall back to a
default.

A calibration dataset is just a **generator function** registered by name —
there are no base classes or dataset objects. The datasets live in
`tensorrt_edgellm/quantization/datasets/` (`builtin.py` for the built-ins,
`__init__.py` for the registry and helpers).

## Built-in datasets

| Modality | Name | Source |
|----------|------|--------|
| Text  | `cnn_dailymail` (default) | `cnn_dailymail` (`3.0.0`, `train`, `article`) |
| Text  | `wikitext` | `Salesforce/wikitext` (`wikitext-2-raw-v1`, `test`, `text`) |
| Image | `mmmu` (default) | `lmms-lab/MMMU` (`validation`, `image`/`image_1`..`image_7` + `question`) |
| Audio | `librispeech` (default) | `openslr/librispeech_asr` (`clean`, `train.100`, `audio` + `text`) |

Text datasets are used for LLM weights, LM head, KV cache, CodePredictor, MTP,
EAGLE3, and DFlash calibration. Image datasets are used when
`--visual_quantization` is set. Audio datasets are used for Qwen3-ASR /
audio-tower calibration.

## The modality contracts

A registered dataset is a zero-argument generator function. The "contract" is
simply what it yields — the calibration loaders bound the sample count,
tokenize, and extract features, so a dataset only produces raw samples:

| Modality | Yields | Notes |
|----------|--------|-------|
| `text`  | `str` | one calibration document/sentence per item |
| `image` | `(image, question)` | `image` is a PIL image, local path, or URL that the model's `AutoProcessor` chat template accepts |
| `audio` | `(audio_bytes, transcript)` | encoded audio (wav/flac/...) decoded by the ASR loader, plus the ground-truth text |

Yield lazily — do not pre-materialise the whole dataset.

## Adding a custom dataset

1. Copy the template
   `tensorrt_edgellm/quantization/datasets/custom_dataset_template.py` to
   `datasets/<your_dataset>.py` (or add a function to `builtin.py`).
2. Write a generator that yields the right type for your modality.
3. Decorate it with `@register("<modality>", "<name>")`. `<name>` is the value
   users pass on the CLI.
4. Register it on import: add `from . import <your_dataset>` to
   `datasets/__init__.py` (next to `from . import builtin`). If you added your
   function to `builtin.py`, this step is already done.

After that, `--<modality>_dataset <name>` works.

```python
from . import load_hf_split, register


@register("text", "my_text")          # users pass: --text_dataset my_text
def my_text():
    ds = load_hf_split("your-org/your-ds", name="my_text", split="train")
    for example in ds:
        text = example.get("text")
        if text:
            yield text                # yield lazily; the loader bounds it
```

When the source is not a Hugging Face dataset (local files, a database,
synthetic prompts, ...), read it yourself in the generator — you do not have
to use `load_hf_split`. See `custom_dataset_template.py` for image and audio
examples.

## Pointing a built-in at a cached / offline copy

Built-in datasets load from the Hugging Face Hub by default. To run on a CI
runner or air-gapped host, point a built-in at a local copy with the
per-dataset environment variable `EDGELLM_QUANT_DATASET_<NAME>` (NAME
upper-cased, with `-` / `/` replaced by `_`). The CLI still passes only the
name:

```bash
export EDGELLM_QUANT_DATASET_CNN_DAILYMAIL=/data/cache/cnn_dailymail
tensorrt-edgellm-quantize llm ... --text_dataset cnn_dailymail
```

The override accepts a JSON/JSONL file, a `datasets.save_to_disk` directory,
or a `load_dataset`-style dataset directory. `load_hf_split` honours it
automatically; custom generators can call `local_override_path(name)`.

## Python API

`quantize_and_export` (and the draft entry points) accept a registered name, a
generator function, or `None` for the modality default:

```python
from tensorrt_edgellm.quantization import quantize_and_export

quantize_and_export(
    model_dir="/path/to/model",
    output_dir="/path/to/output",
    quantization="nvfp4",
    lm_head_quantization="nvfp4",
    text_dataset="cnn_dailymail",            # name, or a generator, or None
)
```

Only the dataset for a modality that the run actually calibrates is resolved,
so an unknown name for an unused modality never fails the run.
