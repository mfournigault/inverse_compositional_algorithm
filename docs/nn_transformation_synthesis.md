# Inverse Compositional Algorithm Repository Analysis — Neural Network Migration Synthesis

## Scope

This document summarizes how to evolve the current inverse compositional algorithm implementation in `mfournigault/inverse_compositional_algorithm`—especially the `src/keras-tf` subdirectory—toward a neural-network-based implementation using **Keras 3 + TensorFlow 2**, while preserving a progressive migration path.

---

## 1) Repository architecture and component map

### Main structure

```text
src/
├── NumPy/Numba reference implementation
│   ├── inverse_compositional_algorithm.py
│   ├── derivatives.py
│   ├── image_optimisation.py
│   ├── transformation.py
│   ├── bicubic_interpolation.py
│   ├── zoom.py
│   └── constants.py
└── keras-tf/ (TensorFlow accelerated implementation)
    ├── tf_inverse_compositional_algorithm.py
    ├── tf_derivatives.py
    ├── tf_image_optimisation.py
    ├── tf_transformation.py
    ├── tf_bicubic_interpolation.py
    └── tf_zoom.py
```

### Entry points and orchestration

- NumPy version:
  - `src/inverse_compositional_algorithm.py:17` `inverse_compositional_algorithm(...)`
  - `src/inverse_compositional_algorithm.py:135` `robust_inverse_compositional_algorithm(...)`
  - `src/inverse_compositional_algorithm.py:264` `pyramidal_inverse_compositional_algorithm(...)`
- TensorFlow/Keras version:
  - `src/keras-tf/tf_inverse_compositional_algorithm.py:61` `class InverseCompositional(Layer)`
  - `src/keras-tf/tf_inverse_compositional_algorithm.py:255` `class RobustInverseCompositional(Layer)`
  - `src/keras-tf/tf_inverse_compositional_algorithm.py:467` `class PyramidalInverseCompositional(Layer)`

### Runtime and dependencies

- NumPy track: `numpy`, `scikit-image`, `scipy`, `numba`
- TF track: Keras + TensorFlow stack (`environment_tf2_gpu_wsl.yml`, including `keras==3.8.0` and `tensorflow==2.18.0`)
- Global constants: `src/constants.py:1-6` (`MAX_ITER`, robust lambda schedule constants, zoom sigma)

---

## 2) Current algorithmic flow and NumPy ↔ TensorFlow mapping

### NumPy reference flow (single-scale quadratic ICA)

In `src/inverse_compositional_algorithm.py:17-133`:

1. Validate image shape/type and initialization
2. Compute template gradients `Ix, Iy` (`:80-93`)
3. Compute analytic Jacobian `J` (`derivatives.py:7-70`)
4. Compute steepest descent images `DIJ` (`image_optimisation.py:158-194`)
5. Compute Hessian `H` and inverse `H_1` (`derivatives.py:73-130`)
6. Iterative loop (`:109-132`):
   - Warp image 2 (`bicubic_interpolation_skimage`)
   - Residual `DI = Iw - I1`
   - Independent vector `b` (`image_optimisation.py:82-110`)
   - Solve `dp = H_1 @ b` (`image_optimisation.py:146-155`)
   - Compose update `p <- p ∘ dp^{-1}` (`transformation.py:35-141`)

### Robust and pyramidal NumPy variants

- Robust variant (`:135-262`) keeps the same skeleton but adds robust weights `rho` (`image_optimisation.py:56-80`), robust `b` and robust `H`.
- Pyramidal variant (`:264-375`) builds image pyramids and propagates parameters across scales using `zoom.zoom_in_parameters` (`zoom.py:62-125`).

### TensorFlow mapping (`src/keras-tf`)

The TensorFlow implementation mirrors the same mechanics:

- Gradients: `tf_derivatives.py:5-25`
- Jacobian: `tf_derivatives.py:35-87`
- Steepest descent images: `tf_image_optimisation.py:72-98`
- Warp and parameter composition:
  - `tf_transformation.py:159-212` (`tf_warp_image`)
  - `tf_transformation.py:216-351` (`tf_update_transform`)
- Main iterative loop with `tf.while_loop`:
  - `InverseCompositional.call` (`tf_inverse_compositional_algorithm.py:137-251`)
  - `RobustInverseCompositional.call` (`:340-465`)
- Pyramidal orchestration:
  - `PyramidalInverseCompositional.call` (`:520-583`)

Important point: classes are instantiated with `trainable=False` (`:114`, `:316`, `:481`). This confirms this is an accelerated analytical algorithm, not yet a trainable neural model.

---

## 3) Analytical, differentiable, iterative blocks and migration potential

### Analytical and differentiable blocks (already tensorized)

1. **Image gradients** (`tf_compute_gradients`) — differentiable finite differences
2. **Geometric Jacobian** (`tf_jacobian`) — closed-form geometry
3. **Steepest descent tensor DIJ** — linear tensor algebra
4. **Hessian and solve** (`H = DIJ^T DIJ`, `dp = H^{-1} b`) — differentiable linear algebra
5. **Warp operator** (`bicubic_sampler`) — differentiable image sampling
6. **Transform update composition** (`tf_update_transform`) — differentiable formulas
7. **Pyramidal parameter scaling** (`tf_zoom_in_parameters`) — deterministic and differentiable

### Numerical constraints / non-smooth points

1. **`tf.linalg.inv(H)` instability** when `H` is singular or poorly conditioned (`tf_inverse_compositional_algorithm.py:189`, `:413`)
2. **NaN boundary masking** introduces hard masks (`mark_boundaries_as_nan`, `:19-37`)
3. **Conditional branching** (`tf.switch_case`, `tf.cond`) for transform/robust type
4. **Variable iteration count** with `tf.while_loop`, less convenient for stable end-to-end training than fixed unrolling
5. **Truncated robust function** has threshold discontinuity (`tf_image_optimisation.py:23`)

---

## 4) Realistic Keras 3 + TensorFlow 2 implementation options

## Option A — Direct parameter prediction network

Predict transformation parameters directly from `(I1, I2)`:

- Backbone CNN + regression head → `p_pred`
- Loss can be supervised (`p_gt`) or photometric (`I1` vs `warp(I2, p_pred)`)

**Pros:** very fast inference (single pass)\
**Cons:** weaker geometric interpretability; may underperform iterative refinement on hard alignments\
**Refactor complexity:** medium

## Option B — Network predicts incremental update per iteration

Replace analytical `dp = H^{-1} b` by learned `dp_net(...)` while keeping warp and composition analytical.

**Pros:** keeps iterative structure, can improve robustness\
**Cons:** needs training data and loop-level integration\
**Refactor complexity:** low-to-medium

## Option C — Hybrid analytical + learned subcomponents

Keep geometric core and learn selected parts:

- learned robust weighting map
- learned preconditioner replacing direct Hessian inversion
- learned gradient/filter front-end

**Pros:** strong compatibility and interpretability\
**Cons:** still requires careful training and numerical safeguards\
**Refactor complexity:** low-to-medium

## Option D — Deep unfolding of fixed ICA iterations

Convert iterative ICA into `N` trainable unfolded steps (shared or unshared weights), preserving analytical operators.

**Pros:** best progressive path; preserves geometry and interpretability; stable training setup\
**Cons:** fixed depth `N` tuning needed\
**Refactor complexity:** low (high reuse of existing `keras-tf` code)

---

## 5) Comparative evaluation

| Criterion | A Direct | B Learned Δp | C Hybrid | D Unfolding |
|---|---|---|---|---|
| Compatibility with current code | Medium | High | Very high | Very high |
| Refactor complexity | Medium | Low/Medium | Low/Medium | Low |
| Training data demand | High | High | Medium | Medium/Low |
| GPU performance | Very high | High | High | High |
| Stability | Variable | Variable | Good | Good |
| Interpretability | Low | Medium | High | Very high |
| Maintenance | Medium | Medium | Good | Good |

---

## 6) Recommended initial strategy

### Recommendation: **Option D (Deep unfolding) first**, with progressive hybridization

Why this is the best first step for this repository:

1. It reuses almost all existing tensorized analytical code in `src/keras-tf`.
2. It avoids full rewrite risk.
3. It preserves existing geometric behavior as a baseline.
4. It enables gradual trainability (start with minimal learnable scalars, then extend).
5. It is naturally compatible with Keras 3 model composition patterns.

This aligns with a **progressive trajectory**, not a disruptive rewrite.

---

## 7) Phased implementation plan (actionable)

## Phase 0 — Baseline refactor without learning

Goal: replicate current ICA behavior with fixed-step unrolling.

Potential new module:

- `src/keras-tf/tf_unfolded_ica.py`
  - `ICAUnfoldedStep(Layer)`
  - `ICAUnfolded(Model)`

Actions:

1. Move one iteration logic from current `body(...)` into `ICAUnfoldedStep`.
2. Build fixed `N`-step forward pass (replace dynamic while convergence loop for training mode).
3. Validate parity against current `InverseCompositional` outputs.

Success criteria:

- Numerical parity on standard test pairs
- No major performance regression

## Phase 1 — Introduce minimal trainable parameters

Goal: add low-risk learnability.

Actions:

1. Add learnable scalar step-size(s) per unfolded iteration.
2. Train with photometric objective using existing warp pipeline.
3. Keep analytical Jacobian and transform composition untouched.

Metrics:

- Final photometric residual
- Parameter error against synthetic ground truth transforms
- Convergence robustness

## Phase 2 — Learn robustness / conditioning

Goal: improve stability in difficult regions and ill-conditioned cases.

Actions:

1. Add learned weighting map or learned preconditioner.
2. Replace or augment direct `H^{-1}` with stabilized/learned equivalent.
3. Keep fallback numerical safeguards.

## Phase 3 — Extend to robust and pyramidal branches

Actions:

1. Integrate unfolded block inside robust branch.
2. Integrate unfolded block at each pyramid scale.
3. Optionally share weights across scales.

## Phase 4 — Keras 3 integration hardening

Actions:

1. Standardize Keras 3 style APIs and ops usage.
2. Add serialization (`get_config`) and clean model interfaces.
3. Add reproducible training/eval scripts.

---

## 8) Training strategy, evaluation, and test plan

### Training data

- Start with synthetic pairs generated from known transforms on existing images (`test/data/*`) to obtain ground-truth parameters.
- Then add unsupervised photometric training on real pairs if available.

### Core losses

- Photometric alignment loss on warped image residual
- Optional parameter regression loss (when `p_gt` is known)
- Optional regularizers on update magnitude or conditioning

### Evaluation metrics

- Parameter error by transform type (translation/euclidean/similarity/affinity/homography)
- Residual photometric error
- Convergence/failure rate
- Throughput and latency on GPU
- Stability under large motion and low-texture areas

### Tests to include

1. Unit tests for each transformed block (warp, update, Jacobian consistency)
2. Regression tests against current TensorFlow analytical implementation
3. Robustness tests for near-singular Hessian scenarios
4. Pyramidal consistency tests across scales

---

## 9) File/component references used in this analysis

- `src/inverse_compositional_algorithm.py` (main NumPy flows: simple/robust/pyramidal)
- `src/derivatives.py` (Jacobian, Hessian, inverse Hessian)
- `src/image_optimisation.py` (robust weights, independent vector, solve, steepest descent images)
- `src/transformation.py` (transform type + composition update formulas)
- `src/zoom.py` (parameter scaling for pyramids)
- `src/constants.py` (iteration and robust scheduling constants)
- `src/keras-tf/tf_inverse_compositional_algorithm.py` (Keras layer implementations)
- `src/keras-tf/tf_derivatives.py` (TF gradients and Jacobian)
- `src/keras-tf/tf_image_optimisation.py` (TF robust and steepest descent ops)
- `src/keras-tf/tf_transformation.py` (TF warp and transform update)
- `src/keras-tf/tf_bicubic_interpolation.py` (differentiable bicubic sampler)
- `src/keras-tf/tf_zoom.py` (TF parameter upscaling)
- `environment_tf2_gpu_wsl.yml` (TF/Keras/GPU runtime)

---

## Final conclusion

The repository already contains a strong tensorized geometric foundation. The most pragmatic and technically coherent migration path toward a neural approach is to **unfold the existing ICA iterations into a trainable Keras model**, then progressively learn only selected components (step size, robustness, conditioning), while preserving the analytical warp and transform composition backbone.

This approach minimizes risk, maximizes reuse, and provides a clear path from current GPU-accelerated analytical optimization to a robust Keras 3 / TensorFlow 2 trainable system.
