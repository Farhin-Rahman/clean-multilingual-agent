# services/edgar.py - SEC EDGAR filings and facts (authoritative, free)

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# SEC EDGAR API configuration
SEC_BASE_URL = "https://www.sec.gov"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinancialAssistant/1.0 (contact@example.com)")
if not os.getenv("SEC_USER_AGENT"):
    logger.warning("SEC_USER_AGENT not set, using default. Set SEC_USER_AGENT='Your Name your@email' for SEC API access")
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT}

class EdgarClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            **SEC_HEADERS,
            "Accept": "application/json",
            "Host": "www.sec.gov"
        })
        self.timeout = 30
        self.rate_limit_delay = 0.2  # 200ms delay between requests
        
    def _make_request(self, url: str) -> Optional[Dict]:
        """Make a polite request to SEC EDGAR API with rate limiting."""
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"SEC API request failed for {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error for {url}: {e}")
            return None
    
    def ticker_to_cik_map(self) -> Dict[str, str]:
        """
        Get mapping from ticker symbols to CIK numbers.
        
        Returns:
            Dict mapping ticker -> CIK
        """
        try:
            url = f"{SEC_BASE_URL}/files/company_tickers.json"
            data = self._make_request(url)
            
            if data:
                ticker_map = {}
                for entry in data.values():
                    if isinstance(entry, dict) and "ticker" in entry and "cik_str" in entry:
                        ticker = entry["ticker"].upper()
                        cik = str(entry["cik_str"]).zfill(10)  # Pad with leading zeros
                        ticker_map[ticker] = cik
                
                logger.info(f"Loaded {len(ticker_map)} ticker-CIK mappings")
                return ticker_map
            
        except Exception as e:
            logger.error(f"Failed to load ticker-CIK map: {e}")
        
        return {}
    
    def get_cik_for_ticker(self, ticker: str) -> Optional[str]:
        """
        Get CIK for a specific ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CIK string or None if not found
        """
        ticker_map = self.ticker_to_cik_map()
        return ticker_map.get(ticker.upper())
    
    def latest_filings_for_ticker(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get latest filings for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of filings to return
            
        Returns:
            List of filing dictionaries
        """
        cik = self.get_cik_for_ticker(ticker)
        if not cik:
            logger.warning(f"No CIK found for ticker {ticker}")
            return []
        
        try:
            url = f"{SEC_BASE_URL}/data/edgar/submissions/CIK{cik}.json"
            data = self._make_request(url)
            
            if not data or "filings" not in data:
                return []
            
            filings = []
            recent_filings = data["filings"]["recent"]
            
            # Extract filing information
            for i in range(min(limit, len(recent_filings["form"]))):
                filing = {
                    "form": recent_filings["form"][i],
                    "filing_date": recent_filings["filingDate"][i],
                    "accession_number": recent_filings["accessionNumber"][i],
                    "file_number": recent_filings["fileNumber"][i] if i < len(recent_filings.get("fileNumber", [])) else None,
                    "items": recent_filings["items"][i] if i < len(recent_filings.get("items", [])) else None,
                    "size": recent_filings["size"][i] if i < len(recent_filings.get("size", [])) else None,
                    "is_xbrl": recent_filings["isXBRL"][i] if i < len(recent_filings.get("isXBRL", [])) else None,
                    "is_inline_xbrl": recent_filings["isInlineXBRL"][i] if i < len(recent_filings.get("isInlineXBRL", [])) else None,
                    "primary_document": recent_filings["primaryDocument"][i] if i < len(recent_filings.get("primaryDocument", [])) else None,
                    "primary_doc_description": recent_filings["primaryDocDescription"][i] if i < len(recent_filings.get("primaryDocDescription", [])) else None
                }
                
                # Only include significant filings
                if filing["form"] in ["10-K", "10-Q", "8-K", "DEF 14A", "S-1", "S-3"]:
                    filings.append(filing)
            
            logger.info(f"Found {len(filings)} significant filings for {ticker}")
            return filings[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get filings for {ticker}: {e}")
            return []
    
    def get_company_facts(self, cik: str) -> Dict[str, Any]:
        """
        Get company facts (XBRL data) for a CIK.
        
        Args:
            cik: Company CIK number
            
        Returns:
            Dict with company facts data
        """
        try:
            url = f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
            data = self._make_request(url)
            
            if data:
                logger.info(f"Retrieved company facts for CIK {cik}")
                return data
            
        except Exception as e:
            logger.error(f"Failed to get company facts for CIK {cik}: {e}")
        
        return {}
    
    def get_company_facts_for_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Get company facts for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict with company facts data
        """
        cik = self.get_cik_for_ticker(ticker)
        if not cik:
            return {}
        
        return self.get_company_facts(cik)
    
    def search_filings(self, ticker: str, form_type: str = None, 
                      start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        Search for specific filings with optional filters.
        
        Args:
            ticker: Stock ticker symbol
            form_type: Form type filter (e.g., '10-K', '10-Q', '8-K')
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            List of matching filings
        """
        filings = self.latest_filings_for_ticker(ticker, limit=50)
        
        if not filings:
            return []
        
        filtered = filings
        
        # Apply form type filter
        if form_type:
            filtered = [f for f in filtered if f["form"] == form_type]
        
        # Apply date filters
        if start_date:
            filtered = [f for f in filtered if f["filing_date"] >= start_date]
        
        if end_date:
            filtered = [f for f in filtered if f["filing_date"] <= end_date]
        
        return filtered
    
    def get_filing_document(self, ticker: str, accession_number: str, 
                           document: str = None) -> Optional[str]:
        """
        Get the text content of a specific filing document.
        
        Args:
            ticker: Stock ticker symbol
            accession_number: Filing accession number
            document: Specific document name (optional)
            
        Returns:
            Document text content or None
        """
        cik = self.get_cik_for_ticker(ticker)
        if not cik:
            return None
        
        try:
            # Clean accession number (remove dashes)
            clean_acc = accession_number.replace("-", "")
            
            if document:
                url = f"{SEC_BASE_URL}/Archives/edgar/data/{cik}/{clean_acc}/{document}"
            else:
                # Try to get the primary document
                filings = self.latest_filings_for_ticker(ticker, limit=20)
                primary_doc = None
                
                for filing in filings:
                    if filing["accession_number"] == accession_number:
                        primary_doc = filing.get("primary_document")
                        break
                
                if not primary_doc:
                    return None
                
                url = f"{SEC_BASE_URL}/Archives/edgar/data/{cik}/{clean_acc}/{primary_doc}"
            
            # Make request for document content
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            return response.text
            
        except Exception as e:
            logger.warning(f"Failed to get document for {ticker} {accession_number}: {e}")
            return None

# Global client instance
_edgar_client = EdgarClient()

# Simple in-process cache for ticker mapping
_ticker_cache = {}
_ticker_cache_timestamp = 0
TICKER_CACHE_TTL = 24 * 3600  # 24 hours

# Convenience functions
def ticker_to_cik_map() -> Dict[str, str]:
    """Get ticker to CIK mapping with caching."""
    import time
    global _ticker_cache, _ticker_cache_timestamp
    
    now = time.time()
    if now - _ticker_cache_timestamp < TICKER_CACHE_TTL and _ticker_cache:
        return _ticker_cache
    
    _ticker_cache = _edgar_client.ticker_to_cik_map()
    _ticker_cache_timestamp = now
    return _ticker_cache

def latest_filings_for_ticker(ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get latest filings for a ticker."""
    return _edgar_client.latest_filings_for_ticker(ticker, limit)

def get_company_facts(cik: str) -> Dict[str, Any]:
    """Get company facts for a CIK."""
    return _edgar_client.get_company_facts(cik)

def get_company_facts_for_ticker(ticker: str) -> Dict[str, Any]:
    """Get company facts for a ticker."""
    return _edgar_client.get_company_facts_for_ticker(ticker)

def search_filings(ticker: str, form_type: str = None, 
                  start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
    """Search filings with filters."""
    return _edgar_client.search_filings(ticker, form_type, start_date, end_date)

def get_filing_document(ticker: str, accession_number: str, document: str = None) -> Optional[str]:
    """Get filing document content."""
    return _edgar_client.get_filing_document(ticker, accession_number, document)
