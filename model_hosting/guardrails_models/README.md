# Guardrails Models Setup

This directory contains the setup for hosting guardrails models using NVIDIA Triton Inference Server. The guardrails models are used for content moderation and safety checks in the ArXiv RAG application.

## Overview

The guardrails system includes three specialized models:
- **Bias Detection**: `valurank/distilroberta-bias` - Detects biased content in text
- **Toxic Comment Detection**: `martin-ha/toxic-comment-model` - Identifies toxic or harmful comments
- **Zero-Shot Classification**: `facebook/bart-large-mnli` - General purpose text classification

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (optional, but recommended for better performance)
- HuggingFace account and token (for private models)

## Setup Instructions

### 1. Environment Variables

Set your HuggingFace token as an environment variable:

```bash
# Windows PowerShell
$env:HF_TOKEN="your_huggingface_token_here"

# Linux/Mac
export HF_TOKEN="your_huggingface_token_here"
```

### 2. Build the Docker Image

From the `model_hosting/guardrails_models` directory:

```bash
docker build -t guardrails-models .
```

### 3. Run the Container

#### Option A: Standalone Container
```bash
docker run -d \
  --name guardrails-triton \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/models:/models \
  guardrails-models
```

#### Option B: With Docker Compose (Recommended)
Add the following service to your main `docker-compose.yml`:

```yaml
services:
  guardrails-models:
    build: ./model_hosting/guardrails_models
    ports:
      - "8000:8000"  # HTTP endpoint
      - "8001:8001"  # gRPC endpoint  
      - "8002:8002"  # Metrics endpoint
    environment:
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ./model_hosting/guardrails_models/models:/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
```

Then run:
```bash
docker-compose up guardrails-models
```

## How It Works

### Model Loading and Conversion Process

1. **Model Loading** (`model_loading.py`):
   - Downloads and loads the transformer models from HuggingFace
   - Validates model accessibility and dependencies
   - Caches models locally for faster subsequent loads

2. **ONNX Conversion** (`onnx_conversion.py`):
   - Converts PyTorch models to ONNX format for optimized inference
   - Creates proper Triton server directory structure:
     ```
     models/
     ├── bart-mnli/
     │   ├── config.pbtxt
     │   └── 1/
     │       ├── model.onnx
     │       └── tokenizer files...
     ├── bias-comment/
     │   ├── config.pbtxt
     │   └── 1/
     │       ├── model.onnx
     │       └── tokenizer files...
     └── toxic-comment/
         ├── config.pbtxt
         └── 1/
             ├── model.onnx
             └── tokenizer files...
     ```

3. **Triton Server Startup** (`run_all.sh`):
   - Executes model loading and conversion scripts
   - Starts NVIDIA Triton Inference Server
   - Provides HTTP and gRPC endpoints for model inference

### Model Configurations

Each model has specific configuration in `config.pbtxt`:

- **Input**: `input_ids` and `attention_mask` (INT64, dynamic dimensions)
- **Output**: `logits` (FP32, model-specific dimensions)
- **Batch Size**: Maximum of 8 requests per batch

## API Usage

Once running, the models are accessible via Triton's REST API:

### Health Check
```bash
curl http://localhost:8000/v2/health/ready
```

### Model Inference
```bash
curl -X POST http://localhost:8000/v2/models/{model_name}/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input_ids",
        "shape": [1, sequence_length],
        "datatype": "INT64",
        "data": [token_ids_array]
      },
      {
        "name": "attention_mask", 
        "shape": [1, sequence_length],
        "datatype": "INT64",
        "data": [attention_mask_array]
      }
    ]
  }'
```

Available model names:
- `bart-mnli` - Zero-shot classification
- `bias-comment` - Bias detection  
- `toxic-comment` - Toxicity detection

## Troubleshooting

### Common Issues

1. **Models fail to download**:
   - Verify your HF_TOKEN is set correctly
   - Check internet connectivity
   - Ensure you have access to private models if applicable

2. **ONNX conversion errors**:
   - Increase Docker memory allocation (recommended: 8GB+)
   - Check model compatibility with optimum library version

3. **Triton server startup fails**:
   - Verify models directory contains proper structure
   - Check that config.pbtxt files are valid
   - Review container logs: `docker logs guardrails-triton`

### Logs and Monitoring

Monitor the container logs:
```bash
docker logs -f guardrails-triton
```

Check Triton metrics:
```bash
curl http://localhost:8002/metrics
```

## Performance Optimization

### Resource Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ for model loading and conversion
- **Storage**: ~5GB for all models in ONNX format
- **GPU**: Optional, enables faster inference

### Scaling Considerations
- Adjust `max_batch_size` in config.pbtxt based on your throughput needs
- Use multiple model instances for high-availability setups
- Consider model quantization for reduced memory usage

## Integration

The guardrails models integrate with the main ArXiv RAG application through:
- Content filtering before processing user queries
- Response validation before returning results
- Real-time toxicity and bias checking

See the main application's guardrails module at `app/agent_infrastructure/guardrails/` for integration examples.

## Development

### Adding New Models

1. Update `MODELS_CONFIG` in `onnx_conversion.py`
2. Add model loading function in `model_loading.py`
3. Rebuild the Docker image
4. Update this documentation

### Testing Models

Use the Triton client libraries to test model functionality:
```python
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")
# Test model inference...
```

## Support

For issues specific to:
- **Model conversion**: Check optimum library documentation
- **Triton server**: Refer to NVIDIA Triton documentation
- **HuggingFace models**: Visit the respective model pages on HuggingFace Hub