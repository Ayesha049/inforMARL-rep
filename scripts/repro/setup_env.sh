#!/bin/bash
# Create the conda env used for the reproduction.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n informarl python=3.10
conda activate informarl
# ignore packages in ~/.local (a stray `multiagent` package there breaks imports)
conda env config vars set PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_modern.txt
python -c "import torch, torch_geometric, gym; print(torch.__version__, torch.cuda.is_available(), torch_geometric.__version__, gym.__version__)"
