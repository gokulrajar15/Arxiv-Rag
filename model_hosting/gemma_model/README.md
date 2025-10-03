# Gemma Embedding Model API

🚀 **Ready for HuggingFace Spaces deployment!**

This is a high-performance embedding API server using a custom fine-tuned Gemma model (`GokulRajaR/embeddinggemma-300m-qat-q8_0-unquantized`) built with LitServe for fast inference and easy deployment.

## 🌟 Features

- **Custom Gemma Model**: Fine-tuned embedding model optimized for query encoding
- **Fast Inference**: Built with LitServe for high-performance serving
- **REST API**: Easy-to-use HTTP endpoints
- **Authentication**: Bearer token security
- **HuggingFace Spaces Ready**: Optimized for seamless deployment

## 🚀 Quick Deploy to HuggingFace Spaces

### Option 1: One-Click Deploy
[![Deploy to HuggingFace Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-to-spaces-sm.svg)](https://huggingface.co/spaces/new?template=https://github.com/your-username/your-repo)

### Option 2: Manual Setup

1. **Create a new Space** on [HuggingFace Spaces](https://huggingface.co/spaces)
2. **Select SDK**: Choose "Docker" 
3. **Upload files**: Copy all files from this directory
4. **Set Environment Variables**:
   - `HF_TOKEN`: Your HuggingFace token (for model access)
   - `auth_token`: API authentication token for clients

### Required Files for Spaces
```
├── Dockerfile
├── requirements.txt  
├── server.py
└── README.md (this file)
```

## 🔧 Local Development

### Prerequisites
- Python 3.11+
- HuggingFace account and token
- Docker (optional)

### Setup

1. **Clone and navigate**:
   ```bash
   git clone <your-repo>
   cd model_hosting/gemma_model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   ```bash
   # Windows PowerShell
   $env:HF_TOKEN="your_huggingface_token"
   $env:auth_token="your_api_auth_token"
   
   # Linux/Mac
   export HF_TOKEN="your_huggingface_token"
   export auth_token="your_api_auth_token"
   ```

4. **Run the server**:
   ```bash
   python server.py
   ```

The server will start on `http://localhost:7860`

### Docker Deployment

```bash
# Build image
docker build -t gemma-embedding-api .

# Run container
docker run -d \
  --name gemma-api \
  -p 7860:7860 \
  -e HF_TOKEN="your_token" \
  -e auth_token="your_auth_token" \
  gemma-embedding-api
```

## 📡 API Usage

### Authentication
All requests require a Bearer token in the Authorization header:

```bash
Authorization: Bearer your_auth_token
```

### Endpoint: Generate Embeddings

**POST** `/predict`

#### Request
```json
{
  "query": "Your text to encode here"
}
```

#### Response
```json
[0.1234, -0.5678, 0.9012, ...]  // 768-dimensional embedding vector
```

### Example Usage

#### cURL
```bash
curl -X POST "http://localhost:7860/predict" \
  -H "Authorization: Bearer your_auth_token" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

#### Python
```python
import requests

url = "http://localhost:7860/predict"
headers = {
    "Authorization": "Bearer your_auth_token",
    "Content-Type": "application/json"
}
data = {"query": "What is machine learning?"}

response = requests.post(url, json=data, headers=headers)
embedding = response.json()
print(f"Embedding dimensions: {len(embedding)}")  # Should output: 768
```

#### JavaScript
```javascript
const response = await fetch('http://localhost:7860/predict', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your_auth_token',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: 'What is machine learning?'
  })
});

const embedding = await response.json();
console.log(`Embedding dimensions: ${embedding.length}`);  // Should output: 768
```

## 🏗️ Architecture

### Model Details
- **Base Model**: Custom Gemma 300M parameters ([Official Gemma Embedding Docs](https://ai.google.dev/gemma/docs/embeddinggemma))
- **Quantization**: Q8_0 quantized but unquantized for deployment
- **Output Dimensions**: 768D embeddings
- **Specialization**: Optimized for query encoding
- **Model Reference**: Based on Google's EmbeddingGemma architecture

### Server Stack
- **Framework**: LitServe (Lightning AI)
- **Backend**: FastAPI
- **Model Loading**: SentenceTransformers
- **Security**: HTTP Bearer token authentication
- **Deployment**: Docker containerized

### Directory Structure
```
gemma_model/
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── server.py              # Main API server
└── README.md              # This documentation
```

## 🔒 Security

- **Authentication**: Bearer token required for all requests
- **Environment Variables**: Sensitive tokens stored as environment variables
- **Model Access**: Private model access via HuggingFace token

## ⚡ Performance

### Specifications
- **Model Size**: ~300M parameters (quantized)
- **Embedding Dimensions**: 768D output vectors
- **Inference Speed**: Optimized with LitServe
- **Memory Usage**: ~2GB RAM recommended
- **CPU/GPU**: CPU optimized (GPU optional)
- **Architecture**: Based on Google's EmbeddingGemma

### Optimization Tips
- Use batch processing for multiple queries
- Consider GPU deployment for higher throughput
- Implement caching for frequently used queries

## 🐛 Troubleshooting

### Common Issues

1. **Model loading fails**:
   ```
   Solution: Verify HF_TOKEN has access to the private model
   ```

2. **Authentication errors**:
   ```
   Solution: Ensure auth_token environment variable is set
   ```

3. **Port conflicts**:
   ```
   Solution: Change port in server.py or use different port mapping
   ```

4. **Memory issues**:
   ```
   Solution: Increase container memory or use quantized version
   ```

### Debug Mode
Enable verbose logging by modifying `server.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🌍 HuggingFace Spaces Configuration

### Recommended Space Settings
- **SDK**: Docker
- **Hardware**: CPU Basic (2 vCPU, 16GB RAM)
- **Visibility**: Private (for API with authentication)
- **Sleep Timeout**: Disable for production use

### Environment Variables in Spaces
Set these in your Space settings:

1. **Go to your Space settings** → **Variables and secrets**
2. **Add the following variables**:
   - `HF_TOKEN`: Your HuggingFace token (for model access)
   - `auth_token`: API authentication token (create a secure random string)

**Setting Environment Variables:**
```
Variable Name: HF_TOKEN
Variable Value: hf_xxxxxxxxxxxxxxxxxxxxxxxxx

Variable Name: auth_token  
Variable Value: your-secure-api-token-here
```

**Note**: Keep your tokens secure and never commit them to your repository!

### Spaces-Specific Dockerfile
The current Dockerfile is already optimized for Spaces:
- Uses port 7860 (Spaces default)
- Proper cache directory setup
- Optimized for container deployment

## 🔄 Integration with ArXiv RAG

This embedding API integrates with the main ArXiv RAG system for:
- **Document Retrieval**: Encoding user queries for similarity search
- **Semantic Search**: Finding relevant ArXiv papers
- **Query Understanding**: Converting natural language to embeddings

### Integration Example
```python
# In your main application
import requests

def get_query_embedding(query, api_url, auth_token):
    response = requests.post(
        f"{api_url}/predict",
        json={"query": query},
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    return response.json()

# Usage
embedding = get_query_embedding(
    "quantum computing applications", 
    "https://your-space.hf.space",
    "your_auth_token"
)
```

## 📈 Monitoring

### Health Check
```bash
curl -H "Authorization: Bearer your_auth_token" \
  http://localhost:7860/health
```

### Metrics
Monitor via logs or integrate with monitoring solutions:
- Request count and latency
- Model inference time
- Memory usage
- Error rates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the embedding API
5. Submit a pull request

## 📄 License

This project is part of the ArXiv RAG system. See the main repository for license details.

## 🆘 Support

- **Issues**: Report bugs in the main repository
- **Model Questions**: Check the [model page](https://huggingface.co/GokulRajaR/embeddinggemma-300m-qat-q8_0-unquantized)
- **LitServe Docs**: [Lightning AI Documentation](https://lightning.ai/docs/litserve)
- **HuggingFace Spaces**: [Spaces Documentation](https://huggingface.co/docs/hub/spaces)

---

⭐ **Ready to deploy?** Click the deploy button above or follow the manual setup guide!