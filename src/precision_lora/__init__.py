"""Precision-preserving LoRA, requiring only PyTorch."""
from .core import enable_lora, prepare_lora, lora_parameters, save_lora, load_lora, merge_lora, LoRALinear
from .graph import capture_lora_step
from .packed import PackedNVFP4Linear
from .calibration import calibrate_packed_layers, hidden_state_evaluator
from .delta import gated_delta_rule

__all__ = ['enable_lora', 'prepare_lora', 'lora_parameters', 'save_lora', 'load_lora',
           'merge_lora', 'LoRALinear', 'capture_lora_step', 'PackedNVFP4Linear', 'calibrate_packed_layers',
           'gated_delta_rule', 'hidden_state_evaluator']
