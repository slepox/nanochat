# CLAUDE.md - nanochat AI Assistant Guide

## Project Overview

nanochat is a full-stack ChatGPT clone implementation designed to run on a single 8XH100 GPU node (~$100-800 budget). Created by Andrej Karpathy as the capstone for LLM101n course.

**Philosophy**: Minimal, readable, hackable codebase. No framework abstractions, config monsters, or model factories.

**Total Size**: ~6,700 lines Python + 475 lines Rust across 45 files

## Technology Stack

- **Python 3.10+** - Main implementation
- **Rust** - Custom BPE tokenizer (rustbpe via PyO3/maturin)
- **PyTorch 2.8+** - Deep learning framework
- **FastAPI/uvicorn** - Web serving
- **uv** - Package manager (fast pip alternative)

### Key Dependencies
```
torch>=2.8.0          # Core framework
fastapi>=0.117.1      # Web server
tiktoken>=0.11.0      # Fast tokenizer inference
tokenizers>=0.22.0    # HuggingFace tokenizer training
datasets>=4.0.0       # Dataset loading
wandb>=0.21.3         # Experiment tracking
maturin>=1.9.4        # Rust-Python bindings (dev)
pytest>=8.0.0         # Testing (dev)
```

## Codebase Structure

```
nanochat/
├── nanochat/                    # Core library (16 modules)
│   ├── gpt.py                   # GPT Transformer model
│   ├── engine.py                # KV-cache inference engine
│   ├── tokenizer.py             # BPE tokenizer wrapper
│   ├── dataloader.py            # Distributed data streaming
│   ├── dataset.py               # Pretraining data utils
│   ├── adamw.py                 # Distributed AdamW optimizer
│   ├── muon.py                  # Distributed Muon optimizer
│   ├── checkpoint_manager.py    # Model save/load
│   ├── common.py                # Utilities & DDP setup
│   ├── configurator.py          # CLI arg parsing alternative
│   ├── precision.py             # Global precision config
│   ├── core_eval.py             # CORE benchmark eval
│   ├── loss_eval.py             # BPB calculation
│   ├── execution.py             # Python code execution tool
│   ├── report.py                # Training report generation
│   └── ui.html                  # Web UI (HTML/JS)
│
├── scripts/                     # Entry points (11 scripts)
│   ├── tok_train.py             # Train tokenizer
│   ├── tok_eval.py              # Evaluate tokenizer
│   ├── base_train.py            # Pretraining
│   ├── base_eval.py             # CORE evaluation
│   ├── base_loss.py             # Validation loss
│   ├── mid_train.py             # Midtraining (conversation format)
│   ├── chat_sft.py              # Supervised fine-tuning
│   ├── chat_rl.py               # Reinforcement learning
│   ├── chat_eval.py             # Chat model evaluation
│   ├── chat_cli.py              # CLI chat interface
│   └── chat_web.py              # FastAPI web server
│
├── tasks/                       # Evaluation benchmarks (8 files)
│   ├── common.py                # Base classes, TaskMixture
│   ├── arc.py                   # ARC science questions
│   ├── gsm8k.py                 # Grade school math
│   ├── mmlu.py                  # Multiple choice
│   ├── humaneval.py             # Python code generation
│   ├── spellingbee.py           # Spelling/counting
│   ├── smoltalk.py              # Dialog dataset
│   └── customjson.py            # Custom JSONL tasks
│
├── rustbpe/                     # Rust tokenizer
│   ├── Cargo.toml
│   └── src/lib.rs               # BPE implementation
│
├── tests/                       # Test suite
│   ├── test_rustbpe.py          # Tokenizer tests
│   ├── test_precision.py
│   └── test_engine.py
│
├── dev/                         # Development utilities
│   ├── gen_synthetic_data.py    # Generate identity data
│   ├── runcpu.sh                # CPU/MPS example
│   └── repackage_data_reference.py
│
├── speedrun.sh                  # $100 tier (~4h on 8XH100)
├── run1000.sh                   # $800 tier (~33h)
├── steprun.sh                   # Step-by-step with auto-detection
├── pyproject.toml               # Project config
└── uv.lock                      # Dependencies lock
```

## Development Setup

### Environment Setup
```bash
# Clone and setup
git clone <repo>
cd nanochat

# Install dependencies (CPU)
uv sync --extra cpu

# Install dependencies (GPU)
uv sync --extra gpu

# Activate virtual environment
source .venv/bin/activate

# Build Rust tokenizer (required)
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Environment Variables
```bash
NANOCHAT_BASE_DIR          # Cache dir (default: ~/.cache/nanochat)
NANOCHAT_PRECISION         # Precision: bfloat16|float16|float32|fp8 (default: bfloat16)
WANDB_RUN                  # W&B run name (default: "dummy")
OMP_NUM_THREADS=1          # Avoid thread conflicts
```

## Common Development Tasks

### Running Tests
```bash
# All tests
python -m pytest tests/ -v -s

# Skip slow tests
python -m pytest tests/ -m "not slow"

# Specific test file
python -m pytest tests/test_rustbpe.py -v -s
```

### Training Pipeline (Full)
```bash
# Quick start - runs entire pipeline
bash speedrun.sh

# Step-by-step execution
bash steprun.sh setup tokenizer base_model
```

### Training Scripts (Individual)

**Tokenizer Training:**
```bash
python -m scripts.tok_train
python -m scripts.tok_eval
```

**Base Model Pretraining:**
```bash
# Multi-GPU (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Single GPU
python -m scripts.base_train --depth=20 --device_batch_size=16

# CPU/MPS (small scale)
python -m scripts.base_train \
  --depth=4 --max_seq_len=512 --device_batch_size=1 \
  --eval_tokens=512 --total_batch_size=512 --num_iterations=20
```

**Finetuning:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

**Evaluation:**
```bash
python -m scripts.base_eval
python -m scripts.chat_eval
```

**Inference:**
```bash
# Web UI
python -m scripts.chat_web

# CLI
python -m scripts.chat_cli
```

## Key Conventions

### Code Style
- **Minimal abstractions** - Functions over heavy OOP
- **DDP-aware logging** - Use `print0()` for rank-0 only output
- **Device agnostic** - Support CUDA, MPS (Mac), CPU with auto-detection
- **No large config objects** - Use `configurator.py` to override globals

### Model Architecture (nanochat/gpt.py)
```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12        # Depth
    n_head: int = 6          # Query heads
    n_kv_head: int = 6       # Key/value heads (MQA)
    n_embd: int = 768        # Embedding dimension
```

**Notable Design Choices:**
- Rotary embeddings (RoPE) instead of positional embeddings
- QK normalization
- ReLU² activation in MLP
- Multi-Query Attention (MQA) for inference efficiency
- No bias in linear layers
- RMSNorm without learnable parameters

### Tokenizer Special Tokens
```python
<|bos|>              # Beginning of sequence
<|user_start|>       # User turn start
<|user_end|>         # User turn end
<|assistant_start|>  # Assistant turn start
<|assistant_end|>    # Assistant turn end
<|python_start|>     # Code execution start
<|python_end|>       # Code execution end
<|output_start|>     # Tool output start
<|output_end|>       # Tool output end
```

### Distributed Training
- Uses PyTorch DDP (Distributed Data Parallel)
- `torchrun --nproc_per_node=8` for 8 GPU training
- Automatic gradient synchronization via NCCL backend
- Gradient accumulation when batch size exceeds memory

### Hyperparameter Tuning
- `--depth` - Number of transformer layers (20 for $100, 26 for $300, 32 for $800)
- `--device_batch_size` - Per-GPU batch size (reduce to fit in VRAM)
- `--total_batch_size` - Effective batch size (auto gradient accumulation)
- `--num_iterations` - Training steps

## Important Patterns

### Configuration via Configurator
Scripts use `configurator.py` to override globals:
```python
# In script:
from nanochat.configurator import override_from_cli
override_from_cli(globals(), prefix='--')

# CLI usage:
python -m scripts.base_train -- --depth=20 --device_batch_size=16
```

### Checkpoint Management
```python
from nanochat.checkpoint_manager import CheckpointManager
ckpt = CheckpointManager(base_dir)
ckpt.save_model(model, step)
model_state = ckpt.load_model(device, step)
```

### Task System
```python
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU

# Combine tasks for training
mixture = TaskMixture([GSM8K(), MMLU()], weights=[0.5, 0.5])
```

### Precision Configuration
```python
from nanochat.precision import get_autocast_ctx, get_precision_dtype

# Use configured precision
with get_autocast_ctx(device):
    output = model(input)
```

## Testing Guidelines

### Test Markers
```python
@pytest.mark.slow  # Skip with -m "not slow"
def test_expensive_operation():
    pass
```

### Test Structure (tests/test_rustbpe.py)
- Correctness tests comparing implementations
- Performance benchmarks (marked slow)
- Interface tests (save/load, encode/decode)
- Integration tests (tiktoken compatibility)

## Common Pitfalls

1. **Memory Issues** - Reduce `--device_batch_size` if OOM
2. **Data Shards** - Ensure enough shards for desired training tokens
3. **Rust Build** - Must run `maturin develop` before using tokenizer
4. **Multi-GPU** - Use `torchrun`, not direct `python` invocation
5. **Environment** - Ensure `source .venv/bin/activate` before running

## File Naming Conventions

- `base_*.py` - Base model (pretraining)
- `mid_*.py` - Midtraining stage
- `chat_*.py` - Chat model (SFT/RL)
- `tok_*.py` - Tokenizer related
- `*_train.py` - Training scripts
- `*_eval.py` - Evaluation scripts

## Important Files for AI Assistants

When modifying this codebase, pay special attention to:

1. **nanochat/gpt.py** - Core model architecture
2. **nanochat/engine.py** - Inference with KV-cache
3. **scripts/base_train.py** - Pretraining logic
4. **tasks/common.py** - Task base classes
5. **pyproject.toml** - Dependencies and build config

## Contributing Guidelines

- **No framework bloat** - Keep it minimal and hackable
- **Declare LLM usage** - Disclose substantial LLM contributions in PRs
- **Maintain readability** - Code should be self-documenting
- **Test critical paths** - Especially tokenizer correctness
- **Device agnostic** - Support CUDA, MPS, CPU

## Quick Reference

**Start training:** `bash speedrun.sh`

**Serve model:** `python -m scripts.chat_web`

**Run tests:** `python -m pytest tests/ -v -s`

**Package codebase:**
```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

**Common model sizes:**
- d20 (561M params) - $100 tier, 4h on 8XH100
- d26 - $300 tier, 12h
- d32 (1.9B params) - $800 tier, 33h
