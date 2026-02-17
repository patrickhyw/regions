# Feature Regions

<img src="sensvis.png" alt="sensitivity_visualization" width="75%">


Exploring how features can be represented as regions instead of directions.

## Design Rationale

### Goals

1. **Full-dimensional** (>0 volume) without needing `~d` points, for generalization to unseen points of the *same* class.
2. **Generative** (as opposed to discriminative) — defined from one class's points only, so regions
  are modular.
3. **Bounded** (finite volume) without needing `~d` points, for generalization to unseen points of *different* classes.
4. **Sensitivity** — correctly includes members.
5. **Specificity** — correctly excludes non-members.

### Types of Regions

| Method | Full-dimensional | Generative | Bounded | Sensitivity | Specificity |
| --- | --- | --- | --- | --- | --- |
| Polytope (linear boundaries) | no | no | no | ~100% | ~100% |
| Convex hull + PCA | no | yes | yes | ~100% | ~100% |
| Hypersphere | yes | yes | yes | ~100% | ~10% |
| **Hyperellipsoid + shrinkage** | **yes** | **yes** | **yes** | **~90%** | **~90%** |

Hyperellipsoids + shrinkage hits the sweet spot across all desired properties.

*repo is wip, adding supporting data soon*