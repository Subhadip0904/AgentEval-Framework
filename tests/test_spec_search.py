# tests/test_spec_search.py
import pytest
from tools.spec_search import spec_search


@pytest.fixture
def nvme_query():
    return "NVMe queue depth"


@pytest.fixture
def ecc_query():
    return "ECC error NAND"


@pytest.fixture
def unknown_query():
    return "XYZ abstract nonsense"


def test_spec_search_nvme(nvme_query):
    """Test spec_search returns correct NVMe documentation"""
    result = spec_search.invoke({"query": nvme_query})
    assert isinstance(result, str)
    assert len(result) > 0
    assert "65535" in result
    assert "NVMe_2.0_spec" in result


def test_spec_search_ecc(ecc_query):
    """Test spec_search returns correct ECC documentation"""
    result = spec_search.invoke({"query": ecc_query})
    assert isinstance(result, str)
    assert len(result) > 0
    assert "NAND_Flash_Guide" in result or "ECC" in result


def test_spec_search_returns_citations(nvme_query):
    """Test that spec_search includes source citations"""
    result = spec_search.invoke({"query": nvme_query})
    assert "[" in result and "]" in result  # Citation format


def test_spec_search_unknown_query(unknown_query):
    """Test spec_search handles unknown queries gracefully"""
    result = spec_search.invoke({"query": unknown_query})
    assert isinstance(result, str)


def test_spec_search_limits_results(nvme_query):
    """Test spec_search returns limited number of results"""
    result = spec_search.invoke({"query": nvme_query})
    # Count "[]" occurrences (each citation is numbered [1], [2], [3])
    citation_count = result.count("[")
    assert citation_count <= 4  # At most top_k=3 results