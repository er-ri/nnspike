import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Union

from nnspike.constants import ROI_CNN
from nnspike.models import NvidiaModel
from nnspike.utils import normalize_image

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest

transform = transforms.ToTensor()


def load_and_prepare_model(model_path, device):
    model = NvidiaModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def process_image(image, device, roi):
    x1, y1, x2, y2 = roi

    roi_area = image[y1:y2, x1:x2]
    roi_area = normalize_image(image=roi_area)
    roi_area = transform(roi_area)
    roi_area = roi_area.to(torch.float32)
    roi_area = roi_area.unsqueeze(0)
    roi_area = roi_area.to(device)

    return roi_area


def create_optimized_model(trained_model, use_qnnpack=True):
    """
    Create an optimized model with quantization and JIT compilation.

    Args:
        trained_model: The trained PyTorch model
        use_qnnpack: Whether to use QNNPACK backend (recommended for ARM processors like Raspberry Pi)

    Returns:
        Optimized model ready for deployment
    """
    trained_model.eval()

    if use_qnnpack:
        # Set QNNPACK as the quantization engine (optimal for ARM64/Raspberry Pi)
        torch.backends.quantized.engine = "qnnpack"

        # For QNNPACK, we need to prepare the model for static quantization
        # This requires a calibration dataset, but for now we'll use dynamic quantization
        # with QNNPACK backend
        quantized_model = torch.quantization.quantize_dynamic(
            trained_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
    else:
        # Default dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            trained_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )

    # JIT compilation for additional performance gains
    scripted_model = torch.jit.script(quantized_model)

    return scripted_model


def create_qnnpack_static_quantized_model(trained_model, calibration_dataloader):
    """
    Create a static quantized model using QNNPACK backend.
    This provides the best performance on ARM processors but requires calibration data.

    Args:
        trained_model: The trained PyTorch model
        calibration_dataloader: DataLoader with representative input data for calibration

    Returns:
        Static quantized model optimized for ARM processors
    """
    # Set QNNPACK as the quantization engine
    torch.backends.quantized.engine = "qnnpack"

    # Prepare model for quantization
    trained_model.eval()
    trained_model.qconfig = torch.quantization.get_default_qconfig("qnnpack")

    # Prepare the model
    prepared_model = torch.quantization.prepare(trained_model)

    # Calibrate with representative data
    with torch.no_grad():
        for roi_area, relative_position in calibration_dataloader:
            prepared_model(roi_area, relative_position)

    # Convert to quantized model
    quantized_model = torch.quantization.convert(prepared_model)

    # JIT compile for additional performance
    scripted_model = torch.jit.script(quantized_model)

    return scripted_model


def load_optimized_model(path, device) -> Union[nn.Module, torch.jit.ScriptModule]:
    """
    Load model with fallback for cross-platform compatibility.
    Tries JIT first (Raspberry Pi optimized), then state_dict (Windows compatible).
    Note: Caller is responsible for NUM_MODES compatibility.
    """
    try:
        # 方法1: JITロード（Raspberry Pi最適化）
        jit_model: torch.jit.ScriptModule = torch.jit.load(path, map_location=device)
        jit_model.eval()
        print("SUCCESS: Model loaded with JIT (optimized for Raspberry Pi)")
        return jit_model
    except Exception as e1:
        print(f"JIT loading failed: {e1}")
        try:
            # 方法2: state_dict読み込み（Windows互換）
            print("Fallback: Loading with state_dict...")
            
            state_dict = torch.load(path, map_location=device)
            
            # NvidiaModelインスタンスを作成（NUM_MODESは呼び出し元で設定済み）
            nvidia_model: NvidiaModel = NvidiaModel()
            nvidia_model.load_state_dict(state_dict)
            nvidia_model = nvidia_model.to(device)
            nvidia_model.eval()
            print("SUCCESS: Model loaded with state_dict (Windows fallback)")
            return nvidia_model
            
        except Exception as e2:
            raise Exception(f"All loading methods failed. JIT: {e1}, State_dict: {e2}")


def model_inference(model, roi_area, tensor_relative_position):
    """
    Perform inference using the model on the given ROI area and relative position.

    Args:
        model: The trained model for inference.
        roi_area: The region of interest image tensor.
        relative_position: The relative position tensor.

    Returns:
        The model's output predictions.
    """
    with torch.no_grad():
        outputs = model(roi_area, tensor_relative_position)

    prob, mode = torch.max(outputs[0], dim=1)
    prob_value = round(prob[0].item(), 2)
    mode_value = mode.item()  # Convert to Python integer
    predicted_x = x1 + (outputs[1][0][0] * (x2 - x1)).detach().item()

    return predicted_x, (mode_value, prob_value)
