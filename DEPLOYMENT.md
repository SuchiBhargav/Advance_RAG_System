# 🚀 Deployment Guide

Complete guide for deploying the Advanced RAG System to production.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB+ for models and data
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Software Requirements
- Python 3.11+
- Docker 20.10+
- Docker Compose 2.0+
- Git

### Optional
- NVIDIA GPU with CUDA support (for faster inference)
- Kubernetes cluster (for production scale)

## Local Development

### 1. Quick Setup

```bash
# Clone repository
git clone <your-repo-url>
cd advanced-rag

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Start the API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Start services
docker-compose up -d qdrant redis ollama

# Pull LLM model
docker exec ollama ollama pull llama3

# Run application
uvicorn src.api.main:app --reload
```

### 3. Access Points

- **API Documentation**: http://localhost:8000/docs
- **API Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Docker Deployment

### Full Stack Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Individual Services

```bash
# Start only specific services
docker-compose up -d qdrant redis

# Restart a service
docker-compose restart rag-api

# Scale API instances
docker-compose up -d --scale rag-api=3
```

### Building Custom Image

```bash
# Build the image
docker build -t advanced-rag:latest .

# Run the container
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  -e QDRANT_HOST=qdrant \
  -e REDIS_HOST=redis \
  --network rag-network \
  advanced-rag:latest
```

## Cloud Deployment

### AWS Deployment

#### Using ECS (Elastic Container Service)

1. **Push Image to ECR**
```bash
# Authenticate to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag advanced-rag:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/advanced-rag:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/advanced-rag:latest
```

2. **Create ECS Task Definition**
```json
{
  "family": "advanced-rag",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "rag-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/advanced-rag:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "QDRANT_HOST", "value": "qdrant.example.com"},
        {"name": "REDIS_HOST", "value": "redis.example.com"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/advanced-rag",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

3. **Create ECS Service**
```bash
aws ecs create-service \
  --cluster production \
  --service-name advanced-rag \
  --task-definition advanced-rag:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

#### Using EC2

```bash
# SSH to EC2 instance
ssh -i key.pem ubuntu@<ec2-ip>

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone and deploy
git clone <repo-url>
cd advanced-rag
docker-compose up -d
```

### Google Cloud Platform (GCP)

#### Using Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/advanced-rag

# Deploy to Cloud Run
gcloud run deploy advanced-rag \
  --image gcr.io/<project-id>/advanced-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

### Azure

#### Using Azure Container Instances

```bash
# Create resource group
az group create --name rag-rg --location eastus

# Create container
az container create \
  --resource-group rag-rg \
  --name advanced-rag \
  --image <registry>.azurecr.io/advanced-rag:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables \
    ENVIRONMENT=production \
    QDRANT_HOST=qdrant.example.com
```

## Kubernetes Deployment

### 1. Create Kubernetes Manifests

**deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: advanced-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: advanced-rag
  template:
    metadata:
      labels:
        app: advanced-rag
    spec:
      containers:
      - name: rag-api
        image: advanced-rag:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: QDRANT_HOST
          value: "qdrant-service"
        - name: REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

**service.yaml**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: advanced-rag-service
spec:
  selector:
    app: advanced-rag
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 2. Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/advanced-rag

# Scale deployment
kubectl scale deployment advanced-rag --replicas=5
```

## Configuration

### Environment Variables

Key environment variables for production:

```bash
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
QDRANT_HOST=qdrant.production.com
QDRANT_PORT=6333
REDIS_HOST=redis.production.com
REDIS_PORT=6379

# Security
RATE_LIMIT_PER_MINUTE=100

# Monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
ENABLE_METRICS=true
```

### Secrets Management

#### Using AWS Secrets Manager

```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Use in application
secrets = get_secret('advanced-rag/production')
```

#### Using Kubernetes Secrets

```bash
# Create secret
kubectl create secret generic rag-secrets \
  --from-literal=langchain-api-key=<key> \
  --from-literal=qdrant-api-key=<key>

# Reference in deployment
env:
- name: LANGCHAIN_API_KEY
  valueFrom:
    secretKeyRef:
      name: rag-secrets
      key: langchain-api-key
```

## Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed metrics
curl http://localhost:8000/metrics
```

### Prometheus Setup

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'advanced-rag'
    static_configs:
      - targets: ['rag-api:9090']
```

### Grafana Dashboards

Access Grafana at http://localhost:3000 (default: admin/admin)

Import dashboard for:
- Request rates
- Response times
- Error rates
- Cache hit rates
- Confidence scores

### Logging

```bash
# View application logs
docker-compose logs -f rag-api

# View specific service logs
docker logs -f <container-id>

# Kubernetes logs
kubectl logs -f deployment/advanced-rag
```

## Troubleshooting

### Common Issues

#### 1. Services Not Starting

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs

# Restart services
docker-compose restart
```

#### 2. Connection Issues

```bash
# Check network
docker network ls
docker network inspect rag-network

# Test connectivity
docker exec rag-api ping qdrant
docker exec rag-api ping redis
```

#### 3. Memory Issues

```bash
# Check resource usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  rag-api:
    deploy:
      resources:
        limits:
          memory: 8G
```

#### 4. Model Loading Issues

```bash
# Check Ollama models
docker exec ollama ollama list

# Pull model again
docker exec ollama ollama pull llama3

# Check model size
docker exec ollama ollama show llama3
```

### Performance Optimization

1. **Enable Caching**: Ensure Redis is properly configured
2. **Adjust Workers**: Increase API_WORKERS for more concurrent requests
3. **Optimize Chunk Size**: Tune CHUNK_SIZE and CHUNK_OVERLAP
4. **Use GPU**: Enable GPU support for faster inference
5. **Scale Horizontally**: Add more API instances

### Support

For issues and questions:
- Check logs: `docker-compose logs -f`
- Review configuration: `.env` file
- Check service health: `/health` endpoint
- Monitor metrics: `/metrics` endpoint

---

**Note**: Always test deployments in a staging environment before production!