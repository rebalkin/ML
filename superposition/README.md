# Toy Models of Superposition — Linear Model

Numerical verification of the analytical results for the **linear model** from the mechanistic interpretability paper:

> Elhage et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread.
> https://transformer-circuits.pub/2022/toy_model/index.html

## Overview

The linear model is a 1-hidden-unit autoencoder with 2 features and sparse inputs. Despite its simplicity, it admits a fully closed-form analysis — making it an ideal sandbox for understanding when and why neural networks represent features in superposition.

The key result: **superposition does not occur in the linear model**. The optimal solution always represents exactly one feature, and the non-trivial critical point θ* is always a saddle, never a minimum.

## Notebook

`linear_superposition_toy_model.ipynb` walks through six verification sections:

| Section | Content |
|:--------|:--------|
| 1. E[L] validation | Closed-form expected loss vs Monte Carlo across random parameter configurations |
| 2. Optimal biases b* | Analytical b*(θ) vs numerical minimisation; relative errors reported |
| 3. Loss landscape & convergence | E[L]\|_{b*}(θ) plotted analytically and via MC at 5k / 50k / 500k samples |
| 4. Dynamic plot | Interactive sliders for I₁, I₂, p₁, p₂ — runs locally only (see note below) |
| 5. Phase diagram (symmetric) | Phase boundary I₂/I₁ = 1 verified for p₁ = p₂ = 0.4 |
| 6. Phase diagram (asymmetric) | Phase boundary I₂/I₁ ≈ 0.316 verified for p₁ = 0.1, p₂ = 0.9 |

The accompanying write-up with full derivations is in `Notes/ML_project.tex`.

## Setup

Dependencies are minimal — no deep learning framework required:

```
numpy
matplotlib
```

Install with:
```bash
pip install numpy matplotlib
```

## Note on the dynamic plot (Section 4)

Section 4 uses `%matplotlib notebook` with `matplotlib.widgets.Slider` and requires running the notebook locally in **classic Jupyter Notebook**. It will not be interactive on GitHub or in JupyterLab without `ipympl`. All other sections render as static output.

## Key analytical results

**Model:** h = wx, x' = w^T h + b, where w = (cos θ, sin θ)

**Expected loss** (derived in closed form):

$$E[\mathcal{L}] = \frac{I_1 p_1 \sin^4\theta}{6} + \frac{I_2 p_2 \cos^4\theta}{6} + \cdots$$

**Optimal biases:**

$$b_i^*(\theta) = -\frac{p_i}{2}(w^\top w)_{ii}$$

**Phase boundary** (exact):

$$\frac{I_2}{I_1} = \frac{p_1(1 - \tfrac{3}{4}p_1)}{p_2(1 - \tfrac{3}{4}p_2)}$$
