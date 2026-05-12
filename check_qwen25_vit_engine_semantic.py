import argparse
import ctypes
import torch
import tensorrt as trt

from tensorrt_edgellm.llm_models.model_utils import load_hf_model
from tensorrt_edgellm.visual_models.qwen2_5_vl_model import Qwen2_5_VisionTransformerPretrainedModelPatch


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    raise TypeError(f"Unhandled TensorRT dtype: {dtype}")


def make_inputs(model, dtype, device, grid_t=1, grid_h=8, grid_w=16):
    hw = grid_t * grid_h * grid_w
    input_tensor = torch.randn(
        (hw, model.config.in_chans * model.config.temporal_patch_size * model.config.patch_size * model.config.patch_size),
        dtype=dtype,
        device=device,
    )
    rotary_pos_emb = torch.randn(
        (hw, model.config.hidden_size // model.config.num_heads // 2),
        dtype=torch.float32,
        device=device,
    )
    attention_mask = torch.zeros((1, hw, hw), dtype=dtype, device=device)
    window_attention_mask = torch.zeros((1, hw, hw), dtype=dtype, device=device)

    window_index = torch.arange(hw // 4, dtype=torch.int64, device=device)
    window_index = window_index.reshape(grid_t, grid_h // 8, 4, grid_w // 8, 4)
    window_index = window_index.permute(0, 1, 3, 2, 4).reshape(-1).contiguous()
    reverse_window_index = torch.argsort(window_index).contiguous()

    return {
        "input": input_tensor.contiguous(),
        "rotary_pos_emb": rotary_pos_emb.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "window_attention_mask": window_attention_mask.contiguous(),
        "window_index": window_index,
        "reverse_window_index": reverse_window_index,
    }


def load_pytorch_visual(model_dir, dtype, device):
    model, _, _ = load_hf_model(model_dir, "fp16", device)
    wrapped = Qwen2_5_VisionTransformerPretrainedModelPatch._from_config(model.visual.config, torch_dtype=dtype)
    wrapped.load_state_dict(model.visual.state_dict())
    wrapped.eval().to(device)
    return wrapped


def run_trt(engine_path, plugin_path, inputs):
    ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    for name, tensor in inputs.items():
        context.set_input_shape(name, tuple(tensor.shape))

    output_name = next(
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
    )

    output = torch.empty(
        tuple(context.get_tensor_shape(output_name)),
        dtype=trt_dtype_to_torch(engine.get_tensor_dtype(output_name)),
        device="cuda",
    )

    for name, tensor in inputs.items():
        context.set_tensor_address(name, tensor.data_ptr())
    context.set_tensor_address(output_name, output.data_ptr())

    stream = torch.cuda.current_stream()
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("TensorRT execute_async_v3 failed")
    stream.synchronize()

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--engine", default="/tmp/qwen25_vit_plugin_engine/visual.engine")
    parser.add_argument("--plugin", default="/workspace/forks/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so")
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    pt_model = load_pytorch_visual(args.model_dir, torch.float16, "cuda")
    inputs = make_inputs(pt_model, torch.float16, "cuda")

    with torch.inference_mode():
        pt_out = pt_model(
            inputs["input"],
            inputs["rotary_pos_emb"],
            inputs["attention_mask"],
            inputs["window_attention_mask"],
            inputs["window_index"],
            inputs["reverse_window_index"],
        ).contiguous()

    trt_out = run_trt(args.engine, args.plugin, inputs).contiguous()

    diff = (pt_out.float() - trt_out.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(pt_out.float().flatten(), trt_out.float().flatten(), dim=0)

    print("PyTorch output shape: ", tuple(pt_out.shape))
    print("TensorRT output shape:", tuple(trt_out.shape))
    print("max_abs_diff:", diff.max().item())
    print("mean_abs_diff:", diff.mean().item())
    print("cosine_similarity:", cosine.item())
    print("PASS: deserialized TensorRT engine ran and was compared against PyTorch.")


if __name__ == "__main__":
    main()
