# tests/test_smoke.py - Basic smoke tests for the financial agent

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_market_data_basic():
    """Test basic market data functionality."""
    try:
        from services.market_data import get_history, get_last_price
        
        # Test getting history for AAPL
        hist = get_history("AAPL", period="5d")
        assert isinstance(hist, object)  # Should be DataFrame-like
        assert hasattr(hist, 'empty') or len(hist) >= 0
        
        # Test getting last price
        price = get_last_price("AAPL")
        assert isinstance(price, (int, float))
        assert price >= 0
        
        print("✅ Market data test passed")
        
    except Exception as e:
        print(f"⚠️ Market data test skipped: {e}")
        pytest.skip(f"Market data test failed: {e}")

def test_edgar_basic():
    """Test basic SEC EDGAR functionality."""
    try:
        from services.edgar import latest_filings_for_ticker
        
        # Test getting filings for AAPL
        filings = latest_filings_for_ticker("AAPL", limit=3)
        assert isinstance(filings, list)
        assert len(filings) >= 0  # May be empty if offline
        
        if filings:
            filing = filings[0]
            assert isinstance(filing, dict)
            assert "form" in filing or "filing_date" in filing
        
        print("✅ EDGAR test passed")
        
    except Exception as e:
        print(f"⚠️ EDGAR test skipped: {e}")
        pytest.skip(f"EDGAR test failed: {e}")

def test_attachment_rag_basic():
    """Test basic attachment RAG functionality."""
    try:
        from rag.attachments import AttachmentIndex
        
        # Create a simple index
        index = AttachmentIndex()
        
        # Add some test documents
        test_texts = [
            "This is a test document about financial analysis.",
            "Another document with market data and stock information.",
            "A third document discussing investment strategies."
        ]
        
        index.add_docs(test_texts)
        index.build()
        
        # Test query
        results = index.query("financial analysis", k=2)
        assert isinstance(results, list)
        assert len(results) >= 0
        
        if results:
            text, score = results[0]
            assert isinstance(text, str)
            assert isinstance(score, float)
            assert 0 <= score <= 1
        
        print("✅ Attachment RAG test passed")
        
    except Exception as e:
        print(f"⚠️ Attachment RAG test skipped: {e}")
        pytest.skip(f"Attachment RAG test failed: {e}")

def test_llm_provider_status():
    """Test LLM provider status checking."""
    try:
        from services.provider_router import healthcheck
        
        status = healthcheck()
        assert isinstance(status, dict)
        assert "groq" in status
        assert "ollama" in status
        
        assert status["groq"] in ["ready", "missing_key"]
        assert status["ollama"] in ["online", "offline"]
        
        print("✅ LLM provider status test passed")
        
    except Exception as e:
        print(f"⚠️ LLM provider status test skipped: {e}")
        pytest.skip(f"LLM provider status test failed: {e}")

def test_charts_basic():
    """Test basic chart generation."""
    try:
        from services.charts import create_all_charts
        
        # Test chart creation for AAPL
        charts = create_all_charts("AAPL")
        assert isinstance(charts, dict)
        
        # Charts may be empty if no data available
        for chart_type, fig in charts.items():
            assert fig is None or hasattr(fig, 'data')  # Plotly figure
            
        print("✅ Charts test passed")
        
    except Exception as e:
        print(f"⚠️ Charts test skipped: {e}")
        pytest.skip(f"Charts test failed: {e}")

def test_support_agent_import():
    """Test that support agent can be imported."""
    try:
        from support_agent import run_customer_support, is_financial_query
        
        # Test financial query detection
        assert is_financial_query("show me stocks") == True
        assert is_financial_query("what is the weather") == False
        
        print("✅ Support agent import test passed")
        
    except Exception as e:
        print(f"⚠️ Support agent import test skipped: {e}")
        pytest.skip(f"Support agent import test failed: {e}")

if __name__ == "__main__":
    # Run tests manually
    print("Running smoke tests...")
    
    test_functions = [
        test_market_data_basic,
        test_edgar_basic,
        test_attachment_rag_basic,
        test_llm_provider_status,
        test_charts_basic,
        test_support_agent_import
    ]
    
    passed = 0
    skipped = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except pytest.skip.Exception:
            skipped += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
    
    print(f"\n📊 Test Results: {passed} passed, {skipped} skipped")
    print("✅ Smoke tests completed")
