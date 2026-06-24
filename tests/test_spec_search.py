# tests/test_spec_search.py
from tools.spec_search import spec_search


def test_nvme_query():
    result = spec_search.invoke({"query": "NVMe queue depth"})
    assert "Queue" in result  # real spec text, not fallback dict


def test_ecc_query():
    result = spec_search.invoke({"query": "ECC error NAND"})
    assert "ECC" in result or "error" in result.lower()