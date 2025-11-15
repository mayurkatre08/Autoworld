# Car RAG System Setup Guide

## Overview
This is a simple Agentic RAG (Retrieval-Augmented Generation) system for car information using:
- **Groq API** (Free tier available) with latest LLaMA model
- **FAISS** for vector similarity search
- **Web scraping** for car data
- **Sentence Transformers** for embeddings

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Free Groq API Key
1. Visit: https://console.groq.com/
2. Sign up for free account
3. Generate API key (free tier: 30 requests/minute)

### 3. Run the System
```bash
python car_rag_system.py
```

## How It Works

### 1. Data Collection
- Scrapes car information from web sources
- Stores structured data (make, model, features, price, etc.)

### 2. Vector Database (FAISS)
- Converts car data to embeddings using SentenceTransformers
- Builds FAISS index for fast similarity search
- Uses cosine similarity for retrieval

### 3. Query Processing
- User query → embedding → FAISS search → relevant cars
- Retrieved cars sent to Groq LLaMA model as context
- Generates natural language response

### 4. Agentic Behavior
- Understands context and intent
- Provides recommendations based on user needs
- Explains reasoning behind suggestions

## Example Queries
- "Show me electric cars under 20 lakhs"
- "Which SUV has the best mileage?"
- "Compare Tata Nexon EV with Hyundai Creta"
- "What are the latest Maruti Suzuki launches?"

## Features
- ✅ Free to use (Groq free tier)
- ✅ Fast vector search with FAISS
- ✅ Latest LLaMA 3.1 70B model
- ✅ Real-time web scraping
- ✅ Interactive chat interface
- ✅ Similarity scoring for transparency

## Architecture
```
User Query → Embedding → FAISS Search → Context → Groq LLaMA → Response
```

## Customization
- Add more car websites in `scrape_car_data()`
- Modify embedding model in `__init__()`
- Adjust search parameters in `search_similar_cars()`
- Change Groq model in `generate_response()`