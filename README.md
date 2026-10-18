# Textonomy: A TnT-LLM-Based Approach for Interpretable Topic Modeling at Scale

Code and experiment configurations accompanying the ICNLSP 2026 paper of the same name.

📄 **Paper:** [ACL Anthology](https://aclanthology.org/2026.icnlsp-1.3/)

**Noah Zuchna**<sup>1,2</sup>, **Martin Stoll**<sup>2</sup>, **Jan Blechschmidt**<sup>2</sup>  
<sup>1</sup>Staffbase &nbsp;&nbsp; <sup>2</sup>Department of Mathematics, Chemnitz University of Technology, Germany  
Contact: n.zuchna@gmail.com

See [Citation](#-citation) for the reference and [Reproducing Experiments](#-reproducing-experiments) to rerun the results.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Easiest way to start:
make quickstart
```

## 📊 Data

Both datasets are available at: https://drive.google.com/drive/folders/1Btw6beim7xY0Tv9lvxYCJ_3fAK8dnyoi

# Python Environment Management
For python env management, uv is being used (for more info refer to https://docs.astral.sh/uv/ or this tutorial https://realpython.com/python-uv/). So, for starters, do this:
```bash
uv sync --all-extras --locked
```
Or use (what is currently the equivalent):
```bash
make install
```

# Troubleshooting:
You can install uv, e.g., using the standalone installer:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
On macos you might have to install dependencies for xgboost:
```bash
brew install libomp
```

## 📁 Repository Structure

```
textonomy/
├── configs/                        # Experiment configurations
│   ├── openai_bills.yaml
│   ├── openai_wiki.yaml
│   ├── oss_bills.yaml
│   └── oss_wiki.yaml
├── data/                           # Data directory
│   ├── bills-hoyle-2022/
│   └── wikitext-103-hoyle-2022/
├── mlartifacts/                    # MLflow artifacts (git ignored)
├── mlruns/                         # MLflow experiment runs (git ignored)
├── notebooks/                      # Jupyter notebooks for experiments, analysis and plotting
├── results/                        # Experiment outputs and results (git ignored)
├── scripts/
│   └── run_experiment.py           # Main script to run experiments
├── src/
│   └── textonomy/                  # Main package source code
├── LICENSE                         # MIT license
├── Makefile                        # Build and utility commands
├── pyproject.toml                  # Project configuration and dependencies
├── uv.lock                         # Lock file for uv package manager
└── README.md                       # This file
```

## 🔬 Reproducing Experiments

To reproduce the experiments, use the `make run-experiment` command. You need to provide the path to a configuration file using the `config` variable.

There are four pre-defined configurations available in the `configs/` directory:

- `configs/openai_bills.yaml`: With OpenAI models on the Bills dataset.
- `configs/openai_wiki.yaml`: With OpenAI models on the Wikitext dataset.
- `configs/oss_bills.yaml`: With Open-source models on the Bills dataset.
- `configs/oss_wiki.yaml`: With Open-source models on the Wikitext dataset.

**Example Usage:**

```bash
# Run experiment with OpenAI on the Bills dataset
make run-experiment config=configs/openai_bills.yaml

# Run experiment with open-source models on the Wikitext dataset
make run-experiment config=configs/oss_wiki.yaml
```

The script `scripts/run_experiment.py` will be executed with the specified configuration, and the results will be logged using MLflow.

## 🔬 Experiment Tracking

This repository uses MLflow for experiment tracking:

1. **Start MLflow UI**: `make mlflow-ui`
2. **View experiments**: Navigate to `http://localhost:8080`
3. **Log experiments**: Use the MLflow wrapper in the notebooks

## 📝 Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{zuchna-etal-2026-textonomy,
    title = "Textonomy: A {T}n{T}-{LLM}-Based Approach for Interpretable Topic Modeling at Scale",
    author = "Zuchna, Noah  and
      Stoll, Martin  and
      Blechschmidt, Jan",
    editor = "Abbas, Mourad  and
      Freihat, Abed Alhakim",
    booktitle = "Proceedings of the 9th International Conference on Natural Language and Speech Processing ({ICNLSP}-2026)",
    month = sep,
    year = "2026",
    address = "University of Trento, Trento, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.icnlsp-1.3/",
    pages = "17--32",
    ISBN = "979-8-89176-458-3"
}
```

## ⚖️ License

Released under the [MIT License](LICENSE).
