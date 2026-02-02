import ast
from pathlib import Path
from typing import List, Set
import pytest

from write_model_class import (
    def_name_and_kind,
    deps_used_names,
    topo_sort_sources,
    dedupe_defs_by_name,
    extract_abs_imports,
    top_level_names,
    find_def_node,
)


def test_def_name_and_kind_class():
    """Test extracting name and kind from class definition."""
    src = "class Foo:\n    pass"
    result = def_name_and_kind(src)
    assert result == ("Foo", "class")


def test_def_name_and_kind_function():
    """Test extracting name and kind from function definition."""
    src = "def bar():\n    pass"
    result = def_name_and_kind(src)
    assert result == ("bar", "function")


def test_def_name_and_kind_async_function():
    """Test extracting name and kind from async function."""
    src = "async def baz():\n    pass"
    result = def_name_and_kind(src)
    assert result == ("baz", "function")


def test_def_name_and_kind_empty():
    """Test handling empty source."""
    src = ""
    result = def_name_and_kind(src)
    assert result is None


def test_def_name_and_kind_import():
    """Test that imports return None."""
    src = "import os"
    result = def_name_and_kind(src)
    assert result is None


def test_deps_used_names_no_dependencies():
    """Test class with no external dependencies."""
    src = "class Foo:\n    def bar(self):\n        return 42"
    avail = {"Foo", "Bar", "Baz"}
    result = deps_used_names(src, avail)
    assert result == set()


def test_deps_used_names_inheritance():
    """Test detecting class inheritance."""
    src = "class Derived(Base):\n    pass"
    avail = {"Base", "Other"}
    result = deps_used_names(src, avail)
    assert result == {"Base"}


def test_deps_used_names_multiple_inheritance():
    """Test detecting multiple inheritance."""
    src = "class Multi(Base1, Base2):\n    pass"
    avail = {"Base1", "Base2", "Base3"}
    result = deps_used_names(src, avail)
    assert result == {"Base1", "Base2"}


def test_deps_used_names_decorator():
    """Test detecting decorator usage."""
    src = "@decorator\nclass Foo:\n    pass"
    avail = {"decorator", "other"}
    result = deps_used_names(src, avail)
    assert result == {"decorator"}


def test_deps_used_names_variable_usage():
    """Test detecting variable usage in class-level attributes (not method bodies)."""
    src = "class Foo:\n    x = Bar()\n    def method(self):\n        return Baz"
    avail = {"Bar", "Baz", "Other"}
    result = deps_used_names(src, avail)
    assert result == {"Bar"}


def test_deps_used_names_filters_unavailable():
    """Test that unavailable names are not included."""
    src = "class Foo(Base):\n    x = Unknown()"
    avail = {"Base", "Other"}
    result = deps_used_names(src, avail)
    assert result == {"Base"}  # Unknown is not in avail


def test_topo_sort_chain_inheritance():
    """Test three-level inheritance chain."""
    base = "class Base:\n    pass"
    mid = "class Middle(Base):\n    pass"
    derived = "class Derived(Middle):\n    pass"
    sources = [derived, base, mid]  # Wrong order

    result = topo_sort_sources(sources)
    names = [def_name_and_kind(s)[0] for s in result]

    base_idx = names.index("Base")
    mid_idx = names.index("Middle")
    derived_idx = names.index("Derived")

    assert base_idx < mid_idx
    assert mid_idx < derived_idx


def test_topo_sort_multiple_inheritance():
    """Test multiple inheritance."""
    base1 = "class Base1:\n    pass"
    base2 = "class Base2:\n    pass"
    derived = "class Derived(Base1, Base2):\n    pass"
    sources = [derived, base2, base1]

    result = topo_sort_sources(sources)
    names = [def_name_and_kind(s)[0] for s in result]

    derived_idx = names.index("Derived")
    base1_idx = names.index("Base1")
    base2_idx = names.index("Base2")

    assert base1_idx < derived_idx
    assert base2_idx < derived_idx


def test_topo_sort_independent_classes():
    """Test independent classes without dependencies."""
    class1 = "class Foo:\n    pass"
    class2 = "class Bar:\n    pass"
    sources = [class1, class2]

    result = topo_sort_sources(sources)
    assert len(result) == 2


def test_topo_sort_complex_dependency_chain():
    """Test the actual use case from the bug report - PairwiseInteraction hierarchy."""
    base = "class PairwiseInteraction:\n    pass"
    gaussian = "class GaussianFusion(PairwiseInteraction):\n    pass"
    equivariant = "class EquivariantInteraction(PairwiseInteraction):\n    pass"
    softmax = "class SoftmaxFusion(EquivariantInteraction):\n    pass"
    other = "class LagrangePolynomial:\n    pass"

    # Simulate wrong order after dedupe
    sources = [equivariant, other, base, gaussian, softmax]

    result = topo_sort_sources(sources)
    names = [def_name_and_kind(s)[0] for s in result]

    pi_idx = names.index("PairwiseInteraction")
    ei_idx = names.index("EquivariantInteraction")
    gf_idx = names.index("GaussianFusion")
    sf_idx = names.index("SoftmaxFusion")

    # PairwiseInteraction must come before its subclasses
    assert pi_idx < ei_idx
    assert pi_idx < gf_idx
    # EquivariantInteraction must come before SoftmaxFusion
    assert ei_idx < sf_idx


def test_topo_sort_with_functions():
    """Test that functions are preserved."""
    func = "def helper():\n    pass"
    class1 = "class Foo:\n    pass"
    sources = [func, class1]

    result = topo_sort_sources(sources)
    assert len(result) == 2


def test_topo_sort_decorator_dependency():
    """Test that decorator dependencies are respected."""
    decorator = "class decorator:\n    pass"
    decorated = "@decorator\nclass Foo:\n    pass"
    sources = [decorated, decorator]

    result = topo_sort_sources(sources)
    names = [def_name_and_kind(s)[0] for s in result]

    dec_idx = names.index("decorator")
    foo_idx = names.index("Foo")
    assert dec_idx < foo_idx


def test_dedupe_no_duplicates():
    """Test deduplication with no duplicates."""
    sources = [
        "class Foo:\n    pass",
        "class Bar:\n    pass",
    ]
    result = dedupe_defs_by_name(sources)
    assert len(result) == 2


def test_dedupe_duplicate_classes():
    """Test deduplication keeps first occurrence."""
    sources = [
        "class Foo:\n    x = 1",
        "class Bar:\n    pass",
        "class Foo:\n    x = 2",  # Duplicate
    ]
    result = dedupe_defs_by_name(sources)
    assert len(result) == 2
    # Should keep first occurrence
    assert "x = 1" in result[0]


def test_dedupe_preserves_order():
    """Test that deduplication preserves order."""
    sources = [
        "class C:\n    pass",
        "class B:\n    pass",
        "class A:\n    pass",
    ]
    result = dedupe_defs_by_name(sources)
    names = [def_name_and_kind(s)[0] for s in result]
    assert names == ["C", "B", "A"]


def test_extract_abs_imports_from_import():
    """Test extracting from import."""
    src = "from pathlib import Path\nclass Foo:\n    pass"
    result = extract_abs_imports(src)
    assert "from pathlib import Path" in result


def test_extract_abs_imports_multiple():
    """Test extracting multiple imports."""
    src = "import os\nimport sys\nfrom pathlib import Path\nclass Foo:\n    pass"
    result = extract_abs_imports(src)
    assert len(result) == 3


def test_top_level_names_classes_and_functions():
    """Test extracting class and function names."""
    src = """
    import os

    class Foo:
        pass

    def bar():
        pass

    class Baz:
        pass
    """
    result = top_level_names(src)
    assert set(result) == {"Foo", "bar", "Baz"}


def test_top_level_names_excludes_imports():
    """Test that imports are excluded."""
    src = "import os\nfrom pathlib import Path\nclass Foo:\n    pass"
    result = top_level_names(src)
    assert result == ["Foo"]


def test_top_level_names_async_functions():
    """Test extracting async function names."""
    src = "async def foo():\n    pass"
    result = top_level_names(src)
    assert result == ["foo"]


def test_find_def_node_class():
    """Test finding a class definition."""
    src = "class Foo:\n    pass\nclass Bar:\n    pass"
    result = find_def_node(src, "Bar")
    assert result is not None
    assert isinstance(result, ast.ClassDef)
    assert result.name == "Bar"


def test_find_def_node_ignores_nested():
    """Test that nested classes are not found."""
    src = """
    class Outer:
        class Inner:
            pass
    """
    result = find_def_node(src, "Inner")
    # Should not find nested classes
    assert result is None


def test_integration_normalizer_hierarchy():
    """Test the actual normalizer hierarchy that was failing."""
    abstract = "class AbstractNormalizer:\n    pass"
    standardize = "class Standardize(AbstractNormalizer):\n    pass"
    log_transform = "class LogTransform(Standardize):\n    pass"
    power_transform = "class PowerTransform(AbstractNormalizer):\n    pass"

    # Wrong order
    sources = [log_transform, standardize, power_transform, abstract]

    result = topo_sort_sources(sources)
    names = [def_name_and_kind(s)[0] for s in result]

    abstract_idx = names.index("AbstractNormalizer")
    standardize_idx = names.index("Standardize")
    log_idx = names.index("LogTransform")
    power_idx = names.index("PowerTransform")

    # AbstractNormalizer must be first
    assert abstract_idx < standardize_idx
    assert abstract_idx < power_idx
    # Standardize before LogTransform
    assert standardize_idx < log_idx


def test_integration_dedupe_and_sort():
    """Test the complete pipeline: dedupe then sort."""
    base = "class Base:\n    pass"
    derived = "class Derived(Base):\n    pass"
    duplicate_base = "class Base:\n    x = 1"

    sources = [derived, duplicate_base, base]

    # Dedupe first (keeps first occurrence)
    deduped = dedupe_defs_by_name(sources)
    assert len(deduped) == 2

    # Then sort
    sorted_sources = topo_sort_sources(deduped)
    names = [def_name_and_kind(s)[0] for s in sorted_sources]

    assert names[0] == "Base"
    assert names[1] == "Derived"


@pytest.mark.parametrize(
    "src,expected_name,expected_kind",
    [
        ("class Foo:\n    pass", "Foo", "class"),
        ("def bar():\n    pass", "bar", "function"),
        ("async def baz():\n    pass", "baz", "function"),
    ],
)
def test_def_name_and_kind_parametrized(src, expected_name, expected_kind):
    """Parametrized test for def_name_and_kind."""
    result = def_name_and_kind(src)
    assert result == (expected_name, expected_kind)


@pytest.mark.parametrize(
    "src,avail,expected",
    [
        ("class Foo:\n    pass", {"Bar"}, set()),
        ("class Derived(Base):\n    pass", {"Base"}, {"Base"}),
        ("class Multi(A, B):\n    pass", {"A", "B", "C"}, {"A", "B"}),
        ("@dec\nclass Foo:\n    pass", {"dec"}, {"dec"}),
    ],
)
def test_deps_used_names_parametrized(src, avail, expected):
    """Parametrized test for deps_used_names."""
    result = deps_used_names(src, avail)
    assert result == expected
