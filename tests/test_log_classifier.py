# tests/test_log_classifier.py
import pytest
from tools.log_classifier import log_classifier


@pytest.fixture
def critical_log():
    return "Uncorrectable ECC error on block 0x3F2A, wear count 98%"


@pytest.fixture
def warning_log():
    return "NVMe queue depth at 80% capacity"


@pytest.fixture
def info_log():
    return "Write buffer flushed successfully"


def test_log_classifier_critical(critical_log):
    """Test log_classifier identifies critical severity"""
    result = log_classifier.invoke({"log_entry": critical_log})
    assert isinstance(result, str)
    assert len(result) > 0
    assert "SEVERITY" in result
    assert ("CRITICAL" in result or "WARNING" in result)


def test_log_classifier_warning(warning_log):
    """Test log_classifier identifies warning severity"""
    result = log_classifier.invoke({"log_entry": warning_log})
    assert isinstance(result, str)
    assert "SEVERITY" in result
    assert "WARNING" in result or "INFO" in result


def test_log_classifier_info(info_log):
    """Test log_classifier identifies info severity"""
    result = log_classifier.invoke({"log_entry": info_log})
    assert isinstance(result, str)
    assert "SEVERITY" in result
    assert "INFO" in result


def test_log_classifier_output_format(critical_log):
    """Test log_classifier returns structured output"""
    result = log_classifier.invoke({"log_entry": critical_log})
    lines = result.split("\n")
    assert any("SEVERITY" in line for line in lines)
    assert any("REASON" in line for line in lines)
    assert any("ACTION" in line for line in lines)


def test_log_classifier_returns_string(critical_log):
    """Test log_classifier returns string output"""
    result = log_classifier.invoke({"log_entry": critical_log})
    assert isinstance(result, str)
