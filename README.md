# Arxiv-RAG: Production-Grade Large Scale AI Agentic RAG Applicat## 🚀 Features

- **Large Scale**: Handles 10M+ indexed documents efficiently
- **AI Agentic**: Intelligent query understanding and multi-step reasoning
- **Multi-Query Retrieval**: Enhanced accuracy through diverse retrieval strategies
- **Production Ready**: Comprehensive monitoring, logging, and error handling
- **✅ Model Hosting**: Complete inference infrastructure with dual engines
  - Triton server for guardrails models (ONNX optimized)
  - LitServe for embedding models (768D vectors)
  - Docker containerization and cloud deployment ready
- **Safety First**: Real-time content filtering, bias detection, and toxicity screening
- **Scalable Architecture**: Microservices design for horizontal scaling
- **Cloud Integration**: HuggingFace Spaces ready deployment
- **Real-time Monitoring**: Complete observability stack with metrics and alertsproduction-ready, large-scale Retrieval-Augmented Generation (RAG) system designed to handle 10M+ indexed documents from ArXiv papers. This application combines AI agents, advanced retrieval mechanisms, and comprehensive monitoring to deliver accurate and contextual responses for scientific research queries.

## 🏗️ Architecture Overview

![Functional Architecture](Functional%20architechture.png)

The system follows a distributed, microservices architecture with the following key components:

### Core Components

- **AI Agent**: Intelligent query processing and response generation
- **Multi-Query Retriever**: Advanced retrieval system with multiple query strategies
- **Embedding Service**: Document vectorization using state-of-the-art embedding models
- **Vector Database**: High-performance storage for 10M+ document embeddings
- **API Layer**: FastAPI-based REST API for system interactions
- **Input Guardrails**: Multi-layered security and validation system including:
  - Toxic detection and content filtering
  - Bias detection mechanisms
  - Topic limiting controls
  - Prompt injection prevention

### Monitoring & Evaluation

- **Agent Monitoring (LangSmith)**: Real-time agent performance tracking and tracing
- **Testing & Evaluation (DeepEval)**: Comprehensive automated evaluation framework
- **Multi-Dimensional Metrics**:
  - **RAG Metrics**: Retrieval accuracy, relevance scoring, and context quality
  - **Agentic Metrics**: Agent decision-making quality and reasoning assessment
  - **Safety Metrics**: Content safety validation and bias detection
  - **Hallucination Metrics**: Response accuracy and factual consistency validation
- **Async Metrics Collection**: Real-time performance data insertion into metrics database
- **Evaluation Setup (DeepEval)**: Automated testing pipeline for continuous quality assurance

### Infrastructure

- **Model Hosting** (✅ **Implemented**): Dual inference engine architecture with complete deployment code
  - **Triton Inference Engine**: High-performance model serving with multiple specialized models:
    - `distill-roberta-bias`: Bias detection model
    - `toxic-comment-model`: Content toxicity detection
    - `bart-large-mnli`: Natural language inference
    - **Location**: `model_hosting/guardrails_models/`
    - **Features**: Docker containerization, ONNX optimization, auto-scaling
  - **LitServe Inference Engine**: Lightweight serving for embedding models:
    - `embeddinggemma-300m`: Primary embedding model (768D vectors)
    - **Location**: `model_hosting/gemma_model/`
    - **Features**: FastAPI integration, HuggingFace Spaces ready, authentication
- **Database**: PostgreSQL for metadata and structured data with async insertion capabilities
- **External Integrations**: Google Colab support for research workflows and dataset management
- **Datasets**: Curated ArXiv paper collection with metadata and continuous ingestion pipeline

## � System Workflow

The system processes queries through a sophisticated multi-stage pipeline:

1. **Input Processing**: User queries are received through the UI and processed by the API layer
2. **Guardrails Validation**: Multi-layered security checks including:
   - Toxic content detection
   - Bias assessment
   - Topic validation
   - Prompt injection prevention
3. **AI Agent Processing**: Intelligent query understanding and planning
4. **Multi-Query Retrieval**: Advanced retrieval with caching mechanisms
5. **Embedding & Vector Search**: Document vectorization and similarity matching
6. **Response Generation**: Context-aware response synthesis
7. **Quality Assurance**: Real-time evaluation and safety checks
8. **Monitoring & Metrics**: Continuous performance tracking and async metrics collection

## �🚀 Features

- **Large Scale**: Handles 10M+ indexed documents efficiently
- **AI Agentic**: Intelligent query understanding and multi-step reasoning
- **Multi-Query Retrieval**: Enhanced accuracy through diverse retrieval strategies
- **Production Ready**: Comprehensive monitoring, logging, and error handling
- **Safety First**: Input validation, content filtering, and hallucination detection
- **Scalable Architecture**: Microservices design for horizontal scaling
- **Real-time Monitoring**: Complete observability stack with metrics and alerts

## 🏗️ Model Hosting Architecture

The system includes complete model hosting infrastructure with two specialized engines:

### Guardrails Models (`model_hosting/guardrails_models/`)
- **Engine**: NVIDIA Triton Inference Server
- **Models**: 3 specialized safety models
- **Format**: ONNX optimized for CPU/GPU inference
- **Features**: 
  - Automatic model conversion and optimization
  - Triton server configuration management
  - Docker containerization with health checks
  - Batch processing support (max batch size: 8)
- **Endpoints**: HTTP (8000), gRPC (8001), Metrics (8002)

### Embedding Model (`model_hosting/gemma_model/`)
- **Engine**: LitServe (Lightning AI)
- **Model**: Custom fine-tuned Gemma embedding model (768D)
- **Features**:
  - FastAPI integration with authentication
  - HuggingFace Spaces ready deployment
  - CPU optimized inference
  - Bearer token security
- **Endpoint**: HTTP (7860)

### Deployment Options
1. **Local Docker**: Complete containerized setup
2. **Docker Compose**: Orchestrated multi-service deployment
3. **HuggingFace Spaces**: Cloud deployment for embedding model
4. **Production**: Kubernetes-ready configurations

## 📊 Performance Metrics

- **Index Size**: 10M+ ArXiv papers
- **Query Response Time**: Sub-second retrieval
- **Embedding Model**: Custom-tuned EmbeddingGemma-300M (768D vectors)
- **Model Hosting**: Production-ready inference engines
- **Concurrent Users**: Supports high-throughput operations
- **Accuracy Metrics**: Comprehensive evaluation framework
- **Safety Models**: Real-time content filtering and bias detection

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python with async capabilities
- **AI/ML Frameworks**: LangChain, LangGraph for agent orchestration
- **Model Serving** (✅ **Implemented**):
  - **Triton Inference Server**: Production-grade model hosting with ONNX optimization
  - **LitServe**: Lightweight inference engine with FastAPI integration
- **Specialized Models** (✅ **Ready for Deployment**):
  - **Embeddings**: `GokulRajaR/embeddinggemma-300m-qat-q8_0-unquantized` (768D vectors)
  - **Safety Models**: `valurank/distilroberta-bias`, `martin-ha/toxic-comment-model`
  - **NLI Model**: `facebook/bart-large-mnli`
- **Containerization**: Docker and Docker Compose for all services
- **Cloud Ready**: HuggingFace Spaces integration for embedding model
- **Vector Database**: High-performance vector storage for embeddings
- **Database**: PostgreSQL with async operations
- **Monitoring & Evaluation**: 
  - LangSmith for agent monitoring
  - DeepEval for automated testing
- **External Integrations**: Google Colab for dataset management

## 🔧 Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL
- OpenAI API keys for text generation models
- HuggingFace account and token
- NVIDIA GPU (optional, recommended for better performance)

### Quick Start

#### 1. Model Hosting Setup

**Guardrails Models (Triton Server)**:
```bash
cd model_hosting/guardrails_models
# Set your HuggingFace token
$env:HF_TOKEN="your_huggingface_token_here"
# Build and run
docker-compose up guardrails-models
```

**Embedding Model (LitServe)**:
```bash
cd model_hosting/gemma_model
# Set environment variables
$env:HF_TOKEN="your_huggingface_token_here"
$env:auth_token="your_api_auth_token"
# Build and run
docker build -t gemma-embedding-api .
docker run -d -p 7860:7860 \
  -e HF_TOKEN=$env:HF_TOKEN \
  -e auth_token=$env:auth_token \
  gemma-embedding-api
```

**Alternative: Deploy to HuggingFace Spaces**
- Navigate to `model_hosting/gemma_model/`
- Follow the README instructions for one-click deployment

#### 2. Main Application Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
$env:OPENAI_API_KEY="your_openai_api_key"
$env:HF_TOKEN="your_huggingface_token"

# Run the application
python -m app.main
```

For detailed setup instructions, see the respective README files in each component directory.

### 📖 Model Hosting Documentation

- **[Guardrails Models Setup](model_hosting/guardrails_models/README.md)** - NVIDIA Triton server setup for safety models
- **[Gemma Embedding Model Setup](model_hosting/gemma_model/README.md)** - LitServe API for embedding model (HuggingFace Spaces ready)

## 🛠️ Technology Stack

## 📋 API Documentation

The system will provide RESTful APIs for:

- Query submission to the RAG system
- System health monitoring
- Performance metrics access
- Evaluation feedback collection

## 🔍 Monitoring & Evaluation

The system includes comprehensive monitoring at multiple levels:

- **Agent Performance**: Query processing efficiency and accuracy
- **Retrieval Quality**: Document relevance and ranking metrics
- **Safety Checks**: Content appropriateness and bias detection
- **System Health**: Infrastructure performance and availability
