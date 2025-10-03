"""
ONNX Model Conversion Script
Converts transformer models to ONNX format for optimized inference.
"""

from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import os

# Configuration
HF_TOKEN = os.getenv('HF_TOKEN')

OUTPUT_DIR = "models"

# Model configurations
MODELS_CONFIG = {
    "bart-mnli": {
        "model_id": "facebook/bart-large-mnli",
        "output_path": "bart-mnli",
        "requires_token": False
    },
    "toxic-comment": {
        "model_id": "martin-ha/toxic-comment-model", 
        "output_path": "toxic-comment",
        "requires_token": True
    },
    "bias-comment": {
        "model_id": "valurank/distilroberta-bias",
        "output_path": "bias-comment",
        "requires_token": True
    }
}

def create_config_pbtxt(model_name, model_path):
    """Create config.pbtxt file for Triton Inference Server based on model type."""
    
    # Define model-specific output dimensions
    output_dims = {
        "bart-mnli": "[ 3 ]",      # 3 classes: entailment, neutral, contradiction
        "toxic-comment": "[ 2 ]",   # 2 classes: non-toxic, toxic
        "bias-comment": "[ 2 ]"     # 2 classes: non-bias, bias
    }
    
    # Get output dimension for this model (default to dynamic if unknown)
    output_dim = output_dims.get(model_name, "[ -1 ]")
    
    config_content = f'''name: "{model_name}"                # must match folder name
platform: "onnxruntime_onnx"     # backend to use
max_batch_size: 8                # max batch requests Triton can handle

input [
  {{
    name: "input_ids"            # must match exported model input
    data_type: TYPE_INT64
    dims: [ -1 ]                 # -1 means dynamic dimension
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]

output [
  {{
    name: "logits"               # must match exported model output
    data_type: TYPE_FP32
    dims: {output_dim}           # model-specific output dimensions
  }}
]

version_policy: {{ latest {{ num_versions: 1 }} }}
'''
    
    config_file = model_path / "config.pbtxt"
    with open(config_file, 'w') as f:
        f.write(config_content)
    print(f"✓ Created config.pbtxt for {model_name}")

def convert_model_to_onnx(model_id, output_path, use_token=False):
    """
    Convert a HuggingFace model to ONNX format in Triton server structure.
    
    Args:
        model_id (str): HuggingFace model identifier
        output_path (str): Path to save the ONNX model
        use_token (bool): Whether to use HF token for private models
    """
    print(f"Converting {model_id} to ONNX...")
    
    # Create directory structure: models/model_name/1/
    model_base_path = Path(OUTPUT_DIR) / output_path
    model_version_path = model_base_path / "1"
    model_version_path.mkdir(parents=True, exist_ok=True)
    
    # Load and convert model
    token = HF_TOKEN if use_token else None
    model = ORTModelForSequenceClassification.from_pretrained(
        model_id, 
        export=True, 
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    
    # Save ONNX model and tokenizer in version directory
    model.save_pretrained(model_version_path)
    tokenizer.save_pretrained(model_version_path)
    
    # Create config.pbtxt in model base directory
    create_config_pbtxt(output_path, model_base_path)
    
    print(f"✓ Model saved to {model_version_path}")
    print(f"✓ Structure: {model_base_path}/{{config.pbtxt, 1/{{model files}}}}")
    return model_base_path

def convert_all_models():
    """Convert all configured models to ONNX format."""
    print("Starting ONNX conversion for all models...\n")
    
    for model_name, config in MODELS_CONFIG.items():
        try:
            convert_model_to_onnx(
                model_id=config["model_id"],
                output_path=config["output_path"], 
                use_token=config["requires_token"]
            )
        except Exception as e:
            print(f"✗ Error converting {model_name}: {e}")
        print()
    
    print("ONNX conversion completed!")

if __name__ == "__main__":
    convert_all_models()