# tests/test_spec_search.py
from tools.spec_search import spec_search

def test_nvme_query():
    result = spec_search.invoke({"query": "NVMe queue depth"})
    assert "65535" in result
    assert "NVMe_2.0_spec" in result

def test_ecc_query():
    result = spec_search.invoke({"query": "ECC error NAND"})
    assert "NAND_Flash_Guide" in result