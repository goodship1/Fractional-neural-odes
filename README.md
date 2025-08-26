# 🧠 TinyNODE: Fractional Neural ODEs with SOE Memory and ESRK Solver

This repo implements a compact, high-performance **Neural ODE with fractional memory** using:

- ✅ **Sum-of-Exponentials (SOE)** to approximate Caputo derivatives
- ✅ Learnable λ and ω parameters for adaptive memory modeling
- ✅ Explicit Stabilized Runge-Kutta (ESRK-15) integration
- ✅ Support for large integration times (e.g. T = 400) with diagnostics

---

## 🔬 Motivation

Many physical systems and sequential learning problems exhibit **long memory**, which standard neural ODEs cannot capture efficiently. We address this by augmenting the ODE with **fractional-order memory dynamics**, implemented efficiently via SOE and trained end-to-end.

---

## 🧠 Model: Fractional ODE with SOE Memory

We model the system as a **fractional differential equation**:

\[
D_t^\alpha x(t) = f(t, x(t))
\]

where \( D_t^\alpha \) is a **Caputo fractional derivative** of order \( \alpha \in (0,1) \):

\[
D_t^\alpha x(t) = \frac{1}{\Gamma(1 - \alpha)} \int_0^t \frac{\dot{x}(s)}{(t - s)^\alpha} \, ds
\]

This models **power-law memory** and non-Markovian dynamics — but is difficult to implement efficiently.

---

## ⚡ SOE Approximation

We approximate the singular kernel \( (t - s)^{-\alpha} \) using a **sum of exponentials** (SOE):

\[
(t - s)^{-\alpha} \approx \sum_{j=1}^m \omega_j e^{-\lambda_j (t - s)}
\]

This converts the fractional integral into an **ODE system** for auxiliary memory states \( Z_j \):

\[
\begin{aligned}
\frac{dZ_j}{dt} &= -\lambda_j Z_j + \omega_j f(t, x) \quad \text{(memory)} \\
\frac{dx}{dt} &= \sum_{j=1}^m Z_j \quad \text{(output)}
\end{aligned}
\]

- \( Z_j \) tracks exponentially decaying memory of the vector field \( f(t, x) \)
- \( \lambda_j, \omega_j \) are **learned** during training (log-parametrized)
- This gives a **learnable approximation to a fractional derivative**, scalable to high dimensions and GPU-friendly.

---

## 🔁 ODE Integration: ESRK-15

We use **ESRK-15**, an explicit stabilized Runge-Kutta method designed for stiff dynamics. It supports:

- ⚖️ High T with small h (or large h with single step)
- 🚀 GPU-friendly stage reuse (two-register scheme)
- 🧪 Stability diagnostics (`Δ-norm`, `h·σ̂` product)

---

## 🧠 Architecture Overview

```text
Input Image (3x32x32)
     ↓
Conv → SiLU → MaxPool
     ↓
Feature map x₀
     ↓
[ODE Block: SOE-Fractional ESRK]
     ↓
Feature map x_T
     ↓
Global Pool + Linear
     ↓
Logits (10 classes)
