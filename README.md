# Feature Regions

<img src="figures/visualize.png" alt="generalization_visualization" width="75%">


Exploring how features can be represented as regions instead of directions.

## Summary

This project studies the geometry of contiguous regions of related features (in text embeddings), finding that hyperellipsoids are a good approximation of feature regions with many desirable properties. It finds that hyperspheres are a poor approximation, which is somewhat surprising given that embeddings are most often compared with cosine similarity.

The project is inspired by [feature splitting](https://transformer-circuits.pub/2023/monosemantic-features) and [spatial structure](https://arxiv.org/abs/2410.19750) and seeks to explicitly model the geometry of these regions. It's related to [hierarchical geometry](https://arxiv.org/abs/2406.14172) but focuses on the geometry of individual categories rather than the structure between categories. It models a different kind of geometry than [non-linear features](https://arxiv.org/abs/2405.14860).

## Design

An ideal region geometry should have all of these properties:

1. **Generative** (as opposed to discriminative) — defined from one class's points only, so regions
  are modular.
2. **Bounded** (finite volume) without needing `O(d)` points, for generalization to unseen points of *different* classes.
3. **Full-dimensional** (>0 volume) without needing `O(d)` points, for generalization to unseen points of the *same* class.
4. **Precision & recall** — to accurately model the shape of the feature.
5. **Simplicity** - always good to have.

Some candidate geometries are:

1. **Linear separation polytope**: a polytope formed by the linear separation boundaries between all classes.
    - Pros: full-dimensional, near perfect precision & recall (almost all classes are linearly separable), simple.
    - Cons: discriminative, not bounded without `O(d)` classes.
2. **Hypersphere**: a hypersphere centered at the mean with radius equal to variance times a confidence threshold.
    - Pros: generative, bounded, full-dimensional, simple.
    - Cons: low precision & recall (see experiments).
3. **Hyperellipsoid + shrinkage**: a hyperellipsoid centered at the mean with radius equal to variance times a confidence threshold, and shrinkage coefficient to regularize the covariance matrix.
    - Pros: generative, bounded, full-dimensional (with shrinkage), high precision & recall (see experiments), simple.
    - Cons: doesn't generalize well to unseen points (see experiments).
4. **Convex hull + tolerance**: a convex hull formed by the points of all classes, with a tolerance parameter to make it full-dimensional.
    - Pros: generative, bounded, full-dimensional (with tolerance), likely high precision & recall due to linear separability.
    - Cons: less simple than hyperellipsoid. Since hyperellipsoids already have great precision & recall, this complexity doesn't seem justified as of now.

## Experiments

### Generalization

Generalization measures how well fitted regions contain held-out members.
Regions are fitted on a training subset of concepts, then tested on
whether the remaining concepts' embeddings fall inside the fitted
region. The train fraction is swept from 0.0 to 0.9.

Each shape (hyperellipsoid, convex hull) is tested with and without
**spaceaug**. With spaceaug, original concepts always go to training,
and whitespace variants (` concept`, `concept `, ` concept `) are
split into train/test. Without spaceaug, the originals themselves
are split.

<img src="figures/generalization.png" alt="generalization" width="75%">

Key results:

- **Hyperellipsoid + spaceaug** dominates, achieving ~0.72–0.98
  accuracy across all train fractions.
- **Hyperellipsoid without spaceaug** is much lower (~0.15–0.50),
  showing that spaceaug provides a large boost.
- **Convex hull + spaceaug** only achieves nonzero accuracy at high
  train fractions.
- **Convex hull without spaceaug** is ~0 everywhere.