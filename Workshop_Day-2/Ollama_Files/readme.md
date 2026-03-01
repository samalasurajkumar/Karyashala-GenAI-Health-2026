# LLM Workshop: RAG + Multi-Agent Healthcare Systems

Welcome to the Hands-On LLM Workshop.

In this workshop, we will build:

- GPT-2 Fine-Tuning for Classification
- Retrieval-Augmented Generation (RAG) System
- LangGraph Multi-Agent Healthcare Risk Assessment System
- PDF-based Knowledge Retrieval
- Ollama-based Local LLM Applications

---

# System Requirements

Recommended:

- Python 3.11
- 16 GB RAM (minimum recommended)
- NVIDIA GPU (optional but recommended)
- 10 GB free disk space
- Windows / Mac / Linux

---

# STEP 1: Install Ollama

Download and install Ollama from:

 https://ollama.com/download

After installation, verify:

```bash
ollama --version
```

---

# STEP 2: Pull Required Models

Open Terminal / Command Prompt and run:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama pull mistral
```

Verify:

```bash
ollama list
```

You should see all three models installed.

---

# STEP 3: Open Your IDE

You may use:

- VS Code
- PyCharm
- Jupyter Notebook
- Anaconda Navigator
- Any Python IDE of your choice

---

# STEP 4: Create Python Environment (Python 3.11 Recommended)

## Option A: Using venv (Standard Python)

Create environment:

```bash
python -m venv llm_env
```

Activate environment:

### Windows
```bash
llm_env\Scripts\activate
```

### Mac / Linux
```bash
source llm_env/bin/activate
```

---

## Option B: Using Conda

Create environment:

```bash
conda create -n llm_env python=3.11
conda activate llm_env
```

---

# STEP 5: Install Required Libraries

---

## 1️ Install GPU-Enabled Core Deep Learning Libraries

If you have NVIDIA GPU with CUDA 12.1:

```bash
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install tensorflow==2.20.0 tiktoken==0.12.0
```

If you DO NOT have GPU:

```bash
pip install torch torchvision torchaudio
pip install tensorflow==2.20.0 tiktoken==0.12.0
```

---

## 2️ Install Orchestration and LLM Libraries

Required for:

- Healthcare Agent
- RAG System
- LangGraph workflows

```bash
pip install langchain langchain-ollama==1.0.1 langgraph==1.0.9 ollama==0.6.1
```

---

## 3️ Install Data Science & PDF Processing Libraries

```bash
pip install numpy==2.3.4 pandas==2.3.3 matplotlib==3.10.7 pymupdf==1.27.1 tqdm==4.67.3
```

---

# STEP 6: Verify Installation

Test GPU availability:

```python
import torch
print(torch.cuda.is_available())
```

If it prints `True`, GPU is working.

Test Ollama:

```bash
ollama run llama3.1:8b
```

If it starts responding, setup is correct.

---

# STEP 7: Run Workshop Notebooks

Open and run the following notebooks:

- GPT-2_Finetuning_Classification.ipynb
- Simple_RAG.ipynb
- LLM_Classifier_LangGraph.ipynb

Run cells sequentially.

---

# Workshop Modules Overview

## 🔹 Module 1: GPT-2 Fine-Tuning
- Text classification
- Tokenization
- Model training
- Evaluation

## 🔹 Module 2: Simple RAG System
- Embeddings using nomic-embed-text
- Vector similarity
- Context retrieval
- Answer generation

## 🔹 Module 3: LangGraph Multi-Agent System
- State-based workflow
- Vitals sanity check agent
- Risk classification agent
- Clinical advice agent

---

# Common Issues & Fixes

## Ollama Not Running?

Start service:

```bash
ollama serve
```

---

## CUDA Errors?

- Ensure NVIDIA drivers installed
- Ensure CUDA compatible
- Install correct PyTorch version

---

## Out of Memory Error?

Use smaller model:

```bash
ollama run mistral
```

Or reduce batch size in training.

---

## pip Installation Errors?

Upgrade pip:

```bash
pip install --upgrade pip
```

---

# After Setup You Can Build

- Local LLM Applications
- RAG-based Chat Systems
- Clinical Triage Systems
- Multi-Agent AI Pipelines
- Custom Fine-Tuned Models

---

# You Are Ready!

If all steps complete successfully, your environment is fully configured for:

- LLM Development
- RAG Systems
- Healthcare AI Prototyping

Happy Building!