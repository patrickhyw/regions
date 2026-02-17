"""3D PCA visualization of bird vs amphibian embeddings."""

import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

from embedding import get_embeddings
from tree import build_named_tree

tree = build_named_tree("animalmin")

bird_node = tree.find("bird")
amphibian_node = tree.find("amphibian")

bird_concepts = bird_node.concepts()
amphibian_concepts = amphibian_node.concepts()
all_concepts = bird_concepts + amphibian_concepts

embeddings = get_embeddings(all_concepts, dimension=768)
X = np.array(embeddings)

pca = PCA(n_components=3)
coords = pca.fit_transform(X)

labels = ["bird"] * len(bird_concepts) + ["amphibian"] * len(amphibian_concepts)

fig = px.scatter_3d(
    x=coords[:, 0],
    y=coords[:, 1],
    z=coords[:, 2],
    color=labels,
    color_discrete_map={"bird": "orange", "amphibian": "green"},
    hover_name=all_concepts,
)
fig.show()
