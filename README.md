# Arxiv-RAG: Production-Grade Large Scale AI Agentic RAG Application

A production-ready, large-scale Retrieval-Augmented Generation (RAG) system designed to handle 10M+ indexed documents from ArXiv papers. This application combines AI agents, advanced retrieval mechanisms, and comprehensive monitoring to deliver accurate and contextual responses for scientific research queries.

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

- **Model Hosting**: Dual inference engine architecture
  - **Triton Inference Engine**: High-performance model serving with multiple specialized models:
    - `distill-roberta-bias`: Bias detection model
    - `toxic-comment-model`: Content toxicity detection
    - `bart-large-mnli`: Natural language inference
  - **LitServe Inference Engine**: Lightweight serving for embedding models:
    - `gamma-embeddings-300m`: Primary embedding model for document vectorization
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

## 📊 Performance Metrics

- **Index Size**: 10M+ ArXiv papers
- **Query Response Time**: Sub-second retrieval
- **Embedding Model**: Custom-tuned embeddinggemma-300m
- **Concurrent Users**: Supports high-throughput operations
- **Accuracy Metrics**: Comprehensive evaluation framework

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python with async capabilities
- **AI/ML Frameworks**: LangChain, LangGraph for agent orchestration
- **Model Serving**:
  - **Triton Inference Server**: Production-grade model hosting
  - **LitServe**: Lightweight inference engine
- **Specialized Models**:
  - **Embeddings**: gamma-embeddings-300m
  - **Safety Models**: distill-roberta-bias, toxic-comment-model
  - **NLI Model**: bart-large-mnli
- **Vector Database**: High-performance vector storage for embeddings
- **Database**: PostgreSQL with async operations
- **Monitoring & Evaluation**: 
  - LangSmith for agent monitoring
  - DeepEval for automated testing
- **External Integrations**: Google Colab for dataset management

## 🔧 Getting Started

### Prerequisites

- Python 3.11
- PostgreSQL
- Openai API keys for text generation models
- Need to host embedding and safety models using Triton and LitServe

### Installation

Setup instructions will be provided as the project develops.

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

## 🤝 Contributing

Contribution guidelines will be established as the project evolves.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ArXiv for providing open access to scientific papers
- The open-source community for the underlying technologies
- Research community for evaluation frameworks and best practices
