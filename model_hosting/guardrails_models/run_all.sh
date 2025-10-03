echo "===== Application Startup at $(date) ====="

echo "Checking Python dependencies..."
python3 -c "import transformers; print(f'transformers version: {transformers.__version__}')" || echo "ERROR: transformers not installed"
python3 -c "import optimum; print(f'optimum version: {optimum.__version__}')" || echo "ERROR: optimum not installed"

echo "Checking models directory..."
if [ ! -d "/models" ]; then
    echo "ERROR: /models directory does not exist"
    exit 1
fi

ls -la /models/
echo "Models directory contents:"
find /models -name "*.pbtxt" -o -name "*.onnx" | head -10

if python3 -c "import transformers, optimum" 2>/dev/null; then
    echo "Running model loading script..."
    python3 model_loading.py
    
    echo "Running ONNX conversion script..."
    python3 onnx_conversion.py
else
    echo "WARNING: Skipping Python scripts due to missing dependencies"
fi

# Start Triton server with better error handling
echo "Starting Triton server..."
tritonserver \
    --model-repository=/models \
    --backend-config=default-max-batch-size=2 \
    --log-verbose=1 \
    --exit-on-error=false
