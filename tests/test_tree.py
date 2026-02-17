import pytest
from pydantic import BaseModel, ValidationError
from pydmodels.knowledge import Concept, KnowledgeNode, KnowledgeTree


class TestKnowledgeTree:
    @pytest.fixture()
    def tree(self) -> KnowledgeTree:
        return KnowledgeTree(
            root=KnowledgeNode(
                concept=Concept("animal"),
                children=[
                    KnowledgeNode(concept=Concept("dog")),
                    KnowledgeNode(
                        concept=Concept("cat"),
                        children=[KnowledgeNode(concept=Concept("persian"))],
                    ),
                ],
            )
        )

    def test_concepts_returns_all(self, tree: KnowledgeTree) -> None:
        assert tree.root.concepts() == [
            Concept("animal"),
            Concept("dog"),
            Concept("cat"),
            Concept("persian"),
        ]

    def test_concepts_single_node(self) -> None:
        tree = KnowledgeTree(root=KnowledgeNode(concept=Concept("root")))
        assert tree.root.concepts() == [Concept("root")]

    def test_unique_concepts(self):
        KnowledgeTree(
            root=KnowledgeNode(
                concept=Concept("animal"),
                children=[
                    KnowledgeNode(concept=Concept("dog"), children=[]),
                    KnowledgeNode(concept=Concept("cat"), children=[]),
                ],
            )
        )

    def test_rejects_duplicate_concepts(self):
        with pytest.raises(ValidationError):
            KnowledgeTree(
                root=KnowledgeNode(
                    concept=Concept("animal"),
                    children=[
                        KnowledgeNode(concept=Concept("dog"), children=[]),
                        KnowledgeNode(concept=Concept("dog"), children=[]),
                    ],
                )
            )

    @pytest.mark.parametrize(
        ("cls", "field"),
        [(KnowledgeNode, "concept"), (KnowledgeTree, "root")],
    )
    def test_frozen(self, cls: type[BaseModel], field: str) -> None:
        if cls is KnowledgeNode:
            instance = KnowledgeNode(concept=Concept("x"))
        else:
            instance = KnowledgeTree(root=KnowledgeNode(concept=Concept("x")))
        with pytest.raises(ValidationError):
            instance.__setattr__(field, "new")

    def test_rejects_duplicate_concepts_nested(self):
        with pytest.raises(ValidationError):
            KnowledgeTree(
                root=KnowledgeNode(
                    concept=Concept("animal"),
                    children=[
                        KnowledgeNode(
                            concept=Concept("dog"),
                            children=[
                                KnowledgeNode(concept=Concept("poodle"), children=[]),
                            ],
                        ),
                        KnowledgeNode(
                            concept=Concept("cat"),
                            children=[
                                KnowledgeNode(concept=Concept("poodle"), children=[]),
                            ],
                        ),
                    ],
                )
            )


import json
from pathlib import Path
from typing import Iterator

import pytest
from treegen.wordnet import build_tree


class _MockSynset:
    """Fake synset that quacks like an NLTK synset and is hashable by lemma names."""

    def __init__(
        self,
        lemma_names: tuple[str, ...],
        hyponyms: list["_MockSynset"] | None = None,
        definition: str = "",
        name: str = "",
    ) -> None:
        self._lemma_names = lemma_names
        self._hyponyms: list[_MockSynset] = list(hyponyms) if hyponyms else []
        self._hypernyms: list[_MockSynset] = []
        self._definition = definition
        self._name = name

    def lemma_names(self) -> tuple[str, ...]:
        return self._lemma_names

    def hyponyms(self) -> list["_MockSynset"]:
        return self._hyponyms

    def hypernyms(self) -> list["_MockSynset"]:
        return self._hypernyms

    def definition(self) -> str:
        return self._definition

    def name(self) -> str:
        return self._name

    def __hash__(self) -> int:
        return hash((self._lemma_names, self._name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MockSynset):
            return NotImplemented
        return (self._lemma_names, self._name) == (
            other._lemma_names,
            other._name,
        )


def fake_synset(
    lemma_names: list[str],
    hyponyms: list[_MockSynset] | None = None,
    definition: str = "",
    name: str = "",
) -> _MockSynset:
    """Return an object that quacks like an NLTK synset."""
    return _MockSynset(tuple(lemma_names), hyponyms, definition=definition, name=name)


def set_hypernyms(root: _MockSynset) -> None:
    """Set the hypernyms attribute of each synset in the root's transitive closure (
    which may be a DAG). The order of hypernyms is based on the preorder traversal."""

    def _traverse(node: _MockSynset) -> Iterator[_MockSynset]:
        yield node
        for child in node.hyponyms():
            yield from _traverse(child)

    all_nodes = list(_traverse(root))
    for node in all_nodes:
        node._hypernyms = []
    for node in all_nodes:
        for child in node.hyponyms():
            child._hypernyms.append(node)


class TestBuildTree:
    """Tests for build_tree using fake synsets."""

    # -- Fixtures ----------------------------------------------------------

    @pytest.fixture()
    def duplicate_bank_root(self) -> _MockSynset:
        """Root with two children whose first lemma is 'bank'."""
        child_a = fake_synset(
            ["bank", "depository"], definition="a financial institution"
        )
        child_b = fake_synset(["bank", "slope"], definition="sloping land")
        root = fake_synset(["root"], hyponyms=[child_a, child_b], definition="top")
        set_hypernyms(root)
        return root

    # -- Concept formatting ------------------------------------------------

    @pytest.mark.parametrize(
        "lemma_names, definition, build_kwargs, expected_concept",
        [
            pytest.param(
                ["leaf"],
                "",
                {},
                "leaf",
                id="single_lemma",
            ),
            pytest.param(
                ["hot_dog"],
                "",
                {},
                "hot dog",
                id="underscore_replacement",
            ),
            pytest.param(
                ["alpha", "beta", "gamma"],
                "",
                {"num_lemmas": 2},
                "alpha, beta",
                id="num_lemmas_joins",
            ),
            pytest.param(
                ["alpha", "beta", "gamma"],
                "",
                {"num_lemmas": "all"},
                "alpha, beta, gamma",
                id="num_lemmas_all",
            ),
            pytest.param(
                ["leaf"],
                "a green thing",
                {"include_definition": True},
                "leaf - a green thing",
                id="include_definition_true",
            ),
            pytest.param(
                ["leaf"],
                "a green thing",
                {"include_definition": False},
                "leaf",
                id="include_definition_false",
            ),
            pytest.param(
                ["alpha", "beta", "gamma"],
                "some definition",
                {"num_lemmas": 2, "include_definition": True},
                "alpha, beta - some definition",
                id="definition_with_num_lemmas",
            ),
        ],
    )
    def test_concept_formatting(
        self,
        lemma_names: list[str],
        definition: str,
        build_kwargs: dict[str, object],
        expected_concept: str,
    ) -> None:
        """Concept string is built from lemma names, num_lemmas, and definition."""
        root = fake_synset(lemma_names, definition=definition)
        set_hypernyms(root)
        tree = build_tree(root, **build_kwargs)
        assert tree.root.concept == expected_concept
        assert tree.root.children == []

    # -- Tree structure ----------------------------------------------------

    def test_hyponyms_become_children(self) -> None:
        """Hyponyms of the root synset become child nodes."""
        child_a = fake_synset(["alpha"])
        child_b = fake_synset(["beta"])
        root = fake_synset(["root"], hyponyms=[child_a, child_b])
        set_hypernyms(root)
        tree = build_tree(root)
        assert len(tree.root.children) == 2
        assert tree.root.children[0].concept == "alpha"
        assert tree.root.children[1].concept == "beta"

    def test_multi_level_tree(self) -> None:
        """Hyponym traversal works recursively across multiple levels."""
        grandchild = fake_synset(["grandchild"])
        child_a = fake_synset(["child_a"], hyponyms=[grandchild])
        child_b = fake_synset(["child_b"])
        root = fake_synset(["root"], hyponyms=[child_a, child_b])
        set_hypernyms(root)
        tree = build_tree(root)
        assert tree.root.concept == "root"
        assert len(tree.root.children) == 2
        assert tree.root.children[0].concept == "child a"
        assert tree.root.children[0].children[0].concept == "grandchild"
        assert tree.root.children[1].children == []

    # -- DAG handling ------------------------------------------------------

    def test_dag_disallow_raises(self) -> None:
        """A shared synset reachable from two parents raises when disallowed."""
        shared = fake_synset(["shared"])
        a = fake_synset(["a"], hyponyms=[shared])
        b = fake_synset(["b"], hyponyms=[shared])
        root = fake_synset(["root"], hyponyms=[a, b])
        set_hypernyms(root)
        with pytest.raises(ValueError, match="DAG detected"):
            build_tree(root, dag_handling="disallow")

    @pytest.mark.parametrize(
        "reverse_hypernyms, expected_shared_parent",
        [(False, "a"), (True, "b")],
        ids=["natural_order", "reversed_order"],
    )
    def test_dag_collapse_assigns_to_primary_parent(
        self, reverse_hypernyms: bool, expected_shared_parent: str
    ) -> None:
        """A shared synset is assigned to its first hypernym."""
        shared = fake_synset(["shared"])
        a = fake_synset(["a"], hyponyms=[shared])
        b = fake_synset(["b"], hyponyms=[shared])
        root = fake_synset(["root"], hyponyms=[a, b])
        # This deterministically makes the order of hypernyms [a, b] since that's the
        # order in the preorder traversal.
        set_hypernyms(root)
        if reverse_hypernyms:
            shared._hypernyms.reverse()
        tree = build_tree(root, dag_handling="collapse")

        subtrees = {c.concept: c for c in tree.root.children}
        other_parent = "b" if expected_shared_parent == "a" else "a"
        assert subtrees[expected_shared_parent].children[0].concept == "shared"
        assert subtrees[other_parent].children == []

    @pytest.mark.parametrize(
        "reverse_hypernyms, expected_shared_parent",
        [(False, "a"), (True, "b")],
        ids=["natural_order", "reversed_order"],
    )
    def test_collapse_preserves_subtree_under_primary_parent(
        self, reverse_hypernyms: bool, expected_shared_parent: str
    ) -> None:
        """Children of a shared node survive under the primary parent."""
        x = fake_synset(["x"])
        y = fake_synset(["y"])
        shared = fake_synset(["shared"], hyponyms=[x, y])
        a = fake_synset(["a"], hyponyms=[shared])
        b = fake_synset(["b"], hyponyms=[shared])
        root = fake_synset(["root"], hyponyms=[a, b])
        set_hypernyms(root)
        if reverse_hypernyms:
            shared._hypernyms.reverse()
        tree = build_tree(root, dag_handling="collapse")

        subtrees = {c.concept: c for c in tree.root.children}
        other_parent = "b" if expected_shared_parent == "a" else "a"
        primary = subtrees[expected_shared_parent]
        assert primary.children[0].concept == "shared"
        assert [c.concept for c in primary.children[0].children] == ["x", "y"]
        assert subtrees[other_parent].children == []

    def test_collapse_handles_two_independent_shared_nodes(self) -> None:
        """Two shared nodes both collapse under the first-visited parent."""
        s1 = fake_synset(["s1"])
        s2 = fake_synset(["s2"])
        a = fake_synset(["a"], hyponyms=[s1, s2])
        b = fake_synset(["b"], hyponyms=[s1, s2])
        root = fake_synset(["root"], hyponyms=[a, b])
        set_hypernyms(root)
        tree = build_tree(root, dag_handling="collapse")

        subtrees = {c.concept: c for c in tree.root.children}
        assert [c.concept for c in subtrees["a"].children] == ["s1", "s2"]
        assert subtrees["b"].children == []
        assert set(tree.root.concepts()) == {"root", "a", "b", "s1", "s2"}

    def test_collapse_handles_triple_parent(self) -> None:
        """A node reachable from three parents collapses under the first."""
        shared = fake_synset(["shared"])
        a = fake_synset(["a"], hyponyms=[shared])
        b = fake_synset(["b"], hyponyms=[shared])
        c = fake_synset(["c"], hyponyms=[shared])
        root = fake_synset(["root"], hyponyms=[a, b, c])
        set_hypernyms(root)
        tree = build_tree(root, dag_handling="collapse")

        subtrees = {ch.concept: ch for ch in tree.root.children}
        assert subtrees["a"].children[0].concept == "shared"
        assert subtrees["b"].children == []
        assert subtrees["c"].children == []

    def test_collapse_handles_nested_diamonds(self) -> None:
        """A diamond within a diamond collapses at both levels."""
        leaf = fake_synset(["leaf"])
        c = fake_synset(["c"], hyponyms=[leaf])
        d = fake_synset(["d"], hyponyms=[leaf])
        mid = fake_synset(["mid"], hyponyms=[c, d])
        a = fake_synset(["a"], hyponyms=[mid])
        b = fake_synset(["b"], hyponyms=[mid])
        root = fake_synset(["root"], hyponyms=[a, b])
        set_hypernyms(root)
        tree = build_tree(root, dag_handling="collapse")

        subtrees = {ch.concept: ch for ch in tree.root.children}
        # Outer diamond: mid stays under a, b becomes leaf.
        assert subtrees["a"].children[0].concept == "mid"
        assert subtrees["b"].children == []
        # Inner diamond: leaf stays under c, d becomes leaf.
        mid_children = {ch.concept: ch for ch in subtrees["a"].children[0].children}
        assert [ch.concept for ch in mid_children["c"].children] == ["leaf"]
        assert mid_children["d"].children == []
        assert set(tree.root.concepts()) == {
            "root",
            "a",
            "b",
            "mid",
            "c",
            "d",
            "leaf",
        }

    def test_collapse_handles_w_shape_dag(self) -> None:
        """Overlapping parent sets assign different shared nodes to different parents."""
        s1 = fake_synset(["s1"])
        s2 = fake_synset(["s2"])
        a = fake_synset(["a"], hyponyms=[s1])
        b = fake_synset(["b"], hyponyms=[s1, s2])
        c = fake_synset(["c"], hyponyms=[s2])
        root = fake_synset(["root"], hyponyms=[a, b, c])
        set_hypernyms(root)
        tree = build_tree(root, dag_handling="collapse")

        subtrees = {ch.concept: ch for ch in tree.root.children}
        # a is primary for s1, b is primary for s2.
        assert [ch.concept for ch in subtrees["a"].children] == ["s1"]
        assert [ch.concept for ch in subtrees["b"].children] == ["s2"]
        assert subtrees["c"].children == []

    # -- Validation / duplicates -------------------------------------------

    @pytest.mark.parametrize("num_lemmas", [0, -1])
    def test_num_lemmas_below_one_raises(self, num_lemmas: int) -> None:
        """num_lemmas < 1 raises ValueError early."""
        root = fake_synset(["leaf"])
        set_hypernyms(root)
        with pytest.raises(ValueError, match="num_lemmas must be >= 1"):
            build_tree(root, num_lemmas=num_lemmas)

    def test_same_first_lemma_raises_by_default(
        self, duplicate_bank_root: _MockSynset
    ) -> None:
        """Two synsets sharing the same first lemma collide at num_lemmas=1."""
        with pytest.raises(ValueError, match="Duplicate concepts"):
            build_tree(duplicate_bank_root)

    @pytest.mark.parametrize(
        "build_kwargs, expected_concepts",
        [
            pytest.param(
                {"num_lemmas": 2},
                {"root", "bank, depository", "bank, slope"},
                id="resolved_by_num_lemmas",
            ),
            pytest.param(
                {"include_definition": True},
                {
                    "root - top",
                    "bank - a financial institution",
                    "bank - sloping land",
                },
                id="resolved_by_definition",
            ),
        ],
    )
    def test_duplicate_first_lemma_resolved(
        self,
        duplicate_bank_root: _MockSynset,
        build_kwargs: dict[str, object],
        expected_concepts: set[str],
    ) -> None:
        """Extra lemmas or definitions disambiguate colliding first lemmas."""
        tree = build_tree(duplicate_bank_root, **build_kwargs)
        assert set(tree.root.concepts()) == expected_concepts

    # -- Duplicate pruning -------------------------------------------------

    @pytest.mark.parametrize(
        "kept_sense, pruned_sense",
        [
            pytest.param("01", "03", id="leading_zero"),
            pytest.param("10", "15", id="no_leading_zero"),
        ],
    )
    def test_prune_keeps_lowest_sense_number(
        self, kept_sense: str, pruned_sense: str
    ) -> None:
        """Pruning keeps the synset with the lowest sense number."""
        savings = fake_synset(["savings"], name="savings.n.01")
        riverbank = fake_synset(["riverbank"], name="riverbank.n.01")
        child_a = fake_synset(["bank"], hyponyms=[savings], name=f"bank.n.{kept_sense}")
        child_b = fake_synset(
            ["bank"], hyponyms=[riverbank], name=f"bank.n.{pruned_sense}"
        )
        root = fake_synset(["root"], hyponyms=[child_a, child_b], name="root.n.01")
        set_hypernyms(root)
        tree = build_tree(root, duplicate_handling="prune")
        assert set(tree.root.concepts()) == {"root", "bank", "savings"}

    def test_prune_nested_duplicate_in_pruned_subtree(self) -> None:
        """A duplicate inside a pruned subtree is removed with it."""
        a = fake_synset(["a"], name="a.n.01")
        x1 = fake_synset(["x"], hyponyms=[a], name="x.n.01")
        d = fake_synset(["d"], name="d.n.01")
        y2 = fake_synset(["y"], hyponyms=[d], name="y.n.02")
        b = fake_synset(["b"], name="b.n.01")
        x2 = fake_synset(["x"], hyponyms=[y2, b], name="x.n.02")
        c = fake_synset(["c"], name="c.n.01")
        y1 = fake_synset(["y"], hyponyms=[c], name="y.n.01")
        root = fake_synset(["root"], hyponyms=[x1, x2, y1], name="root.n.01")
        set_hypernyms(root)
        tree = build_tree(root, duplicate_handling="prune")
        assert set(tree.root.concepts()) == {"root", "x", "a", "y", "c"}

    def test_prune_duplicates_all_at_once(self) -> None:
        """Slight overpruning due to all at once pruning is intentional;
        see the docstring of build_tree for more details."""
        a = fake_synset(["a"], name="a.n.01")
        x1 = fake_synset(["x"], hyponyms=[a], name="x.n.01")
        d = fake_synset(["d"], name="d.n.01")
        y2 = fake_synset(["y"], hyponyms=[d], name="y.n.01")
        b = fake_synset(["b"], name="b.n.01")
        x2 = fake_synset(["x"], hyponyms=[y2, b], name="x.n.02")
        c = fake_synset(["c"], name="c.n.01")
        y1 = fake_synset(["y"], hyponyms=[c], name="y.n.02")
        root = fake_synset(["root"], hyponyms=[x1, x2, y1], name="root.n.01")
        set_hypernyms(root)
        tree = build_tree(root, duplicate_handling="prune")
        # y.n.02 and x.n.02 should both be pruned (the former also causing
        # y.n.01 to become unreachable). This slightly overprunes but is
        # intentional; see the docstring of build_tree for more details.
        assert set(tree.root.concepts()) == {"root", "x", "a"}

    # -- Replace -----------------------------------------------------------

    @pytest.fixture()
    def replace_root(self) -> _MockSynset:
        """Tree: root -> [mid -> [deep_a, deep_b], child_a, child_b]."""
        deep_a = fake_synset(["deep a"])
        deep_b = fake_synset(["deep b"])
        mid = fake_synset(["mid"], hyponyms=[deep_a, deep_b])
        child_a = fake_synset(["child a"])
        child_b = fake_synset(["child b"])
        root = fake_synset(["root"], hyponyms=[mid, child_a, child_b])
        set_hypernyms(root)
        return root

    def test_replace_renames_concept(self, replace_root: _MockSynset) -> None:
        """A str value in replace renames the concept."""
        tree = build_tree(replace_root, replace={"child a": "renamed"})
        assert "renamed" in tree.root.concepts()
        assert "child a" not in tree.root.concepts()

    def test_replace_removes_and_promotes_children(
        self, replace_root: _MockSynset
    ) -> None:
        """A None value removes the node and promotes its children."""
        tree = build_tree(replace_root, replace={"mid": None})
        child_concepts = [c.concept for c in tree.root.children]
        assert "mid" not in child_concepts
        assert "deep a" in child_concepts
        assert "deep b" in child_concepts

    def test_replace_missing_key_ignored(self, replace_root: _MockSynset) -> None:
        """A key not present in the tree is silently ignored."""
        tree = build_tree(replace_root, replace={"nonexistent": "x"})
        assert set(tree.root.concepts()) == {
            "root",
            "mid",
            "deep a",
            "deep b",
            "child a",
            "child b",
        }

    def test_replace_root_none_raises(self, replace_root: _MockSynset) -> None:
        """Mapping the root to None raises ValueError."""
        with pytest.raises(ValueError, match="Cannot remove the root"):
            build_tree(replace_root, replace={"root": None})

    def test_replace_leaf_none_removes(self, replace_root: _MockSynset) -> None:
        """A leaf mapped to None simply disappears."""
        tree = build_tree(replace_root, replace={"child a": None})
        assert "child a" not in tree.root.concepts()

    def test_replace_chained_none(self) -> None:
        """Two levels of None removal promotes grandchildren to root."""
        gc = fake_synset(["gc"])
        mid = fake_synset(["mid"], hyponyms=[gc])
        outer = fake_synset(["outer"], hyponyms=[mid])
        root = fake_synset(["root"], hyponyms=[outer])
        set_hypernyms(root)
        tree = build_tree(root, replace={"outer": None, "mid": None})
        assert set(tree.root.concepts()) == {"root", "gc"}
        assert tree.root.children[0].concept == "gc"

    def test_replace_creating_duplicate_raises(self) -> None:
        """Renaming to collide with an existing concept raises ValueError."""
        a = fake_synset(["a"])
        b = fake_synset(["b"])
        root = fake_synset(["root"], hyponyms=[a, b])
        set_hypernyms(root)
        with pytest.raises(ValueError, match="Duplicate concepts"):
            build_tree(root, replace={"a": "b"})

    def test_replace_applied_before_pruning(self) -> None:
        """Replace removals promote children before pruning runs."""
        c = fake_synset(["c"], name="c.n.01")
        a1 = fake_synset(["a"], name="a.n.01")
        a2 = fake_synset(["a"], hyponyms=[c], name="a.n.02")
        root = fake_synset(["root"], hyponyms=[a1, a2], name="root.n.01")
        set_hypernyms(root)
        # Both a1 and a2 have concept "a" which maps to None. a2's
        # child c is promoted to root before pruning runs, so c
        # survives.
        tree = build_tree(
            root,
            duplicate_handling="prune",
            replace={"a": None},
        )
        assert set(tree.root.concepts()) == {"root", "c"}


class TestAnimalReplaceWhenMinimalJson:
    @pytest.fixture()
    def data(self) -> dict[str, str | None]:
        path = Path(__file__).parent.parent / "animal_replace_when_minimal.json"
        with open(path) as f:
            return json.load(f)

    def test_all_values_are_str_or_none(self, data: dict[str, str | None]) -> None:
        for key, value in data.items():
            assert isinstance(value, (str, type(None))), (
                f"Key {key!r} has value of type {type(value).__name__},"
                f" expected str or None"
            )

    def test_loaded_dict_matches_module_constant(
        self, data: dict[str, str | None]
    ) -> None:
        from tree import ANIMAL_REPLACE_WHEN_MINIMAL

        assert data == ANIMAL_REPLACE_WHEN_MINIMAL
