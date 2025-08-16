# tests/test_core.py — minimal correctness tests
import re
from support_agent import validate_sql_ast, numeric_verification, mcdm_score, apply_rules

def test_sql_validation_allows_select_companies():
    q = "SELECT symbol, name FROM companies LIMIT 5"
    out = validate_sql_ast(q)
    assert out.lower().startswith("select")
    assert "companies" in out

def test_sql_validation_blocks_non_select():
    import pytest
    with pytest.raises(ValueError):
        validate_sql_ast("DELETE FROM portfolios")

def test_numeric_verification_pe_consistency():
    c = {"price": 100.0, "pe_ratio": 20.0, "eps": 5.0}
    vr = numeric_verification(c)
    assert vr["numeric_ok"] is True

def test_numeric_verification_negative_pe_flags():
    c = {"price": 50.0, "pe_ratio": -10.0}
    vr = numeric_verification(c)
    assert vr["numeric_ok"] is False

def test_mcdm_score_bounds():
    c = {"pe_ratio": 15, "beta": 0.9, "volatility": 0.25, "price": 100, "sma_20": 95, "rsi": 55,
         "market_cap": 50e9, "sentiment": 0.1}
    weights = {"valuation":0.25,"risk":0.30,"momentum":0.20,"size":0.15,"sentiment":0.10}
    s, feats = mcdm_score(c, weights)
    assert 0.0 <= s <= 1.0
    assert set(feats.keys()) == {"valuation","risk","momentum","size","sentiment"}

def test_rules_block_high_beta_for_conservative():
    company = {"beta": 1.6}
    profile = {"risk_tolerance": "Conservative"}
    out = apply_rules(company, profile)
    assert out["blocked"] is True
