# Pattern Recognition

This repository contains study materials and assignments for Pattern Recognition, with a focus on Bayes-based classification.

## Repository structure (por trabalho)

| Trabalho | Notebook(s) | LaTeX / PDF | Figuras |
|----------|-------------|-------------|---------|
| **1** | `Bayes/Trabalho1_SolucaoCompleta.ipynb` | `documents/main.tex` → `documents/main.pdf` | `documents/figures/` |
| **2** | `Bayes/Trabalho2_ReconhecimentoPadroes.ipynb` | *(a criar)* `documents/trabalho2/main.tex` | `documents/trabalho2/figures/` |

Outros ficheiros de estudo: `Bayes/RecPadroesBayes.ipynb`, `Bayes/bayes.py`.

## Requirements

- Python 3.x (for running scripts/notebooks)
- Jupyter Notebook or JupyterLab (optional, for `.ipynb` files)
- A LaTeX distribution (e.g., MiKTeX or TeX Live) to compile `.tex` files

## How to Use

### Run Python code

From the repository root:

```bash
python Bayes/bayes.py
```

### Open a notebook

```bash
jupyter notebook Bayes/Trabalho1_SolucaoCompleta.ipynb
jupyter notebook Bayes/Trabalho2_ReconhecimentoPadroes.ipynb
```

### Compile the LaTeX document

From the `documents/` folder:

```bash
pdflatex main.tex
```

This generates `main.pdf`.

## License

This project is distributed under the terms of the license in `LICENSE`.