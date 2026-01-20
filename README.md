# MIMIC-ED Deep Learning Imputation Project

Deep learning imputation for missing values in MIMIC-ED medical data.

## Setup

### Install uv
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Initialize Project
```bash
git clone https://github.com/alicoppe/deep-learning-imputation.git
cd mimic-ed-imputation
uv sync
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### Configure Paths
```bash
cp .env.example .env
# Edit .env with your data paths
```

### Add Data
Place MIMIC-ED files in data/mimic/ed/

## Structure
```text
├── configs/              # Experiment configurations
├── data/                 # Data (gitignored)
│   ├── mimic/ed/        # Raw MIMIC-ED data
│   └── processed/       # Preprocessed data
├── models/checkpoints/  # Model weights (gitignored)
├── notebooks/           # Personal notebooks per student
├── src/                 # Shared source code
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # Model architectures
│   ├── training/       # Training & evaluation
│   └── utils/          # Utilities
└── pyproject.toml      # Dependencies
```

## Usage

### Working in Notebooks

```python
import sys
sys.path.append('../..')

from src.data.loader import load_mimic_ed_data
from src.models.imputation_model import ImputationModel
```

### Package Management
```bash
uv add <package>          # Add dependency
uv remove <package>       # Remove dependency
uv sync                   # Sync after git pull
```

### Verify PyTorch in the venv
```bash
uv run -- python -m src.utils.torch_hello
python src/utils/torch_hello.py
```

## Best Practices

- Never commit data or model checkpoints
- Test in notebooks, move reusable code to src/
- Pull and sync before working: git pull && uv sync
