# Feature Regions

<img src="specvis.png" alt="sensitivity_visualization" width="75%">


Exploring how features can be represented as regions instead of directions.

## Design Rationale

### Goals

1. **Generative** (as opposed to discriminative) — defined from one class's points only, so regions
  are modular.
2. **Bounded** (finite volume) without needing `~d` points, for generalization to unseen points of *different* classes.
3. **Full-dimensional** (>0 volume) without needing `~d` points, for generalization to unseen points of the *same* class.
4. **Sensitivity** — correctly includes members.
5. **Specificity** — correctly excludes non-members.

### Types of Regions

| Method | Generative | Bounded | Full-dimensional | Sensitivity | Specificity |
| --- | --- | --- | --- | --- | --- |
| Polytope (linear boundaries) | no | no | no | ~100% | ~100% |
| Convex hull + PCA | yes | yes | no | ~100% | ~100% |
| Hypersphere | yes | yes | yes | ~100% | ~10% |
| **Hyperellipsoid + shrinkage** | **yes** | **yes** | **yes** | **~90%** | **~90%** |

Hyperellipsoids + shrinkage hits the sweet spot across all desired properties.

*repo is wip, adding supporting data soon*