# Feature Regions

Exploring how features can be represented as regions instead of directions.

## Design Rationale

### Goals

- **Specificity and sensitivity** — correctly include members and
  exclude non-members.
- **Full-dimensional** (>0 volume) with as few as 2 points, so new
  points aren't trivially outside the region.
- **Bounded** (finite volume) for generalization to unseen points.
- **Sample-efficient** — needs far fewer points than dimensions, since
  most WordNet concepts have fewer than 3,072 examples.
- **Efficient** in high dimensions.
- **Generative** — defined from one class's points only, so regions
  are modular and stable as labels change.
- **Simple.**

### Why Hyperellipsoids with Shrinkage

| | Full-dim with ≥2 points | Bounded | Generative | Efficient | Specificity |
|---|---|---|---|---|---|
| Polytope (linear boundaries) | — | needs ≥d+1 categories | no | — | high |
| Convex hull | needs ≥d+1 points | yes | yes | expensive | high |
| Hypersphere | yes | yes | yes | cheap | ~10% |
| **Hyperellipsoid + shrinkage** | **yes** | **yes** | **yes** | **cheap** | **~90%** |

The hyperellipsoid with shrinkage hits the sweet spot across all
goals, achieving ~90% specificity where a hypersphere manages only
~10%.
