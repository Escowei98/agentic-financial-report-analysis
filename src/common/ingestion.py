"""
SEC EDGAR 10-K Filing Ingestion Pipeline.

Downloads 10-K filings from SEC EDGAR using a three-layer architecture:
  Layer 1 — Discovery:  edgartools for filing metadata & accession numbers
  Layer 2 — Download:   edgartools → datamule/secsgml → direct httpx (with retry)
  Layer 3 — Parse:      edgartools TenK → regex fills gaps → full text fallback

Output format: Markdown (.md) + metadata sidecar (.meta.json)
- Markdown: Structured text with headers for semantic chunking (RAG)
  and token-efficient full-context input (Long Context).
- JSON sidecar: Machine-readable metadata (ticker, CIK, dates).

Shared by ALL 4 systems for fair comparison.
"""

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from edgar import Company, set_identity
from edgar.httpclient import configure_http
from lxml import html as lxml_html
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.common.config import load_config

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# 10-K section header patterns for regex-based fallback extraction.
# `FINANCIAL\s*STATE\s*MENTS?` tolerates intra-word whitespace between "STATE"
# and "MENTS" as seen in EDGAR text-conversions of MSFT FY2023/FY2024 filings
# (e.g. "FINANCIAL STATE MENTS AND SUPPLEMENTARY DATA").
SECTION_PATTERNS = {
    "Business": r"(?:ITEM\s*1[\.:\s]|ITEM\s*1\b)(?!\d)\s*[\-—–]?\s*(?:BUSINESS)",
    "Risk Factors": r"ITEM\s*1A[\.:\s]\s*[\-—–]?\s*(?:RISK\s*FACTORS)",
    "MD&A": r"ITEM\s*7[\.:\s]\s*[\-—–]?\s*(?:MANAGEMENT.S?\s*DISCUSSION)",
    "Financial Statements": r"ITEM\s*8[\.:\s]\s*[\-—–]?\s*(?:FINANCIAL\s*STATE\s*MENTS?)",
    "Directors and Corporate Governance": r"ITEM\s*10[\.:\s]\s*[\-—–]?\s*(?:DIRECTORS)",
}

# Direct attributes on the TenK object (most reliable)
TENK_DIRECT_ATTRS = {
    "business": "Business",
    "risk_factors": "Risk Factors",
    "management_discussion": "MD&A",
    "directors_officers_and_governance": "Directors and Corporate Governance",
}

# Item-based lookups via TenK.__getitem__ for sections without direct properties.
# Routed through edgartools' `sections` parser (with fallback to `chunked_document`
# and `CrossReferenceIndex`), which is more robust than the regex fallback.
TENK_ITEM_LOOKUPS = {
    "Item 8": "Financial Statements",
}


# ---------------------------------------------------------------------------
#  Config-driven settings
# ---------------------------------------------------------------------------

def _get_sec_config() -> dict:
    """Load SEC EDGAR config from base.yaml."""
    config = load_config()
    return config.get("sec_edgar", {})


def _get_rate_limit() -> float:
    """Get rate limit delay from config (seconds between SEC requests)."""
    return _get_sec_config().get("rate_limit_seconds", 0.15)


def _get_http_timeout() -> int:
    """Get HTTP timeout from config."""
    return _get_sec_config().get("http_timeout", 120)


def _get_retry_config() -> dict:
    """Get retry configuration from config."""
    sec = _get_sec_config()
    return {
        "max_retries": sec.get("max_retries", 3),
        "backoff_min": sec.get("backoff_min", 5),
        "backoff_max": sec.get("backoff_max", 60),
    }


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class FilingMetadata:
    """Metadata for a downloaded 10-K filing."""

    ticker: str
    company_name: str
    cik: str
    filing_date: str
    accession_number: str
    fiscal_year_end: str
    form_type: str = "10-K"


@dataclass
class ProcessedFiling:
    """A processed 10-K filing with extracted sections."""

    metadata: FilingMetadata
    sections: dict[str, str] = field(default_factory=dict)
    full_text: str = ""

    def to_markdown(self) -> str:
        """
        Convert filing to clean Markdown format.

        Structure:
        - YAML-style header with key metadata
        - Each 10-K section as a ## heading
        - Clean text content under each heading
        """
        lines = []

        # Document title
        m = self.metadata
        lines.append(f"# {m.company_name} — 10-K Annual Report")
        lines.append("")
        lines.append(f"**Ticker:** {m.ticker}  ")
        lines.append(f"**CIK:** {m.cik}  ")
        lines.append(f"**Filing Date:** {m.filing_date}  ")
        lines.append(f"**Fiscal Year End:** {m.fiscal_year_end}  ")
        lines.append(f"**Accession Number:** {m.accession_number}")
        lines.append("")
        lines.append("---")
        lines.append("")

        if self.sections:
            for section_name, content in self.sections.items():
                lines.append(f"## {section_name}")
                lines.append("")
                lines.append(content)
                lines.append("")
        else:
            # Fallback: full text without section headers
            lines.append("## Full Filing Text")
            lines.append("")
            lines.append(self.full_text)

        return "\n".join(lines)

    def to_metadata_dict(self) -> dict:
        """Return metadata as dict for JSON sidecar."""
        return {
            **asdict(self.metadata),
            "num_sections": len(self.sections),
            "section_names": list(self.sections.keys()),
            "total_chars": len(self.full_text),
        }

    @classmethod
    def from_files(cls, md_path: Path, meta_path: Path) -> "ProcessedFiling":
        """Load from Markdown + metadata JSON sidecar."""
        # Load metadata
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata = FilingMetadata(
            ticker=meta["ticker"],
            company_name=meta["company_name"],
            cik=meta["cik"],
            filing_date=meta["filing_date"],
            accession_number=meta["accession_number"],
            fiscal_year_end=meta["fiscal_year_end"],
            form_type=meta.get("form_type", "10-K"),
        )

        # Load markdown and parse sections
        full_text = md_path.read_text(encoding="utf-8")
        sections = _parse_markdown_sections(full_text)

        return cls(metadata=metadata, sections=sections, full_text=full_text)


# ---------------------------------------------------------------------------
#  EDGAR initialization
# ---------------------------------------------------------------------------

_edgar_initialized = False


def _init_edgar() -> None:
    """Initialize edgartools with SEC-required identity and robust HTTP settings."""
    global _edgar_initialized

    config = load_config()
    email = config.get("sec_edgar_email")
    if not email:
        raise ValueError(
            "SEC_EDGAR_EMAIL not set. SEC requires identification. "
            "Set it in your .env file."
        )

    if not _edgar_initialized:
        timeout = _get_http_timeout()
        # Set CRAWL mode for conservative access
        os.environ["EDGAR_ACCESS_MODE"] = "CRAWL"
        set_identity(email)
        configure_http(timeout=timeout)
        _edgar_initialized = True
        logger.info("EDGAR initialized: identity=%s, timeout=%ds, mode=CRAWL", email, timeout)


def _get_sec_email() -> str:
    """Get the SEC email from config."""
    config = load_config()
    return config.get("sec_edgar_email", "anonymous@example.com")


# ---------------------------------------------------------------------------
#  Content type detection
# ---------------------------------------------------------------------------

def _detect_content_type(content: str) -> str:
    """
    Detect whether content is HTML, SGML, or plain text.

    Returns: 'html', 'sgml', or 'text'
    """
    # Check first 2000 chars for markers
    head = content[:2000].upper()

    # SGML markers used by SEC filings
    sgml_markers = ["<DOCUMENT>", "<TYPE>", "<SEQUENCE>", "<SEC-DOCUMENT>"]
    if any(marker in head for marker in sgml_markers):
        return "sgml"

    # HTML markers
    if "<HTML" in head or "<!DOCTYPE" in head or "<TABLE" in head:
        return "html"

    return "text"


# ---------------------------------------------------------------------------
#  SGML handling (via datamule/secsgml)
# ---------------------------------------------------------------------------

def _extract_primary_doc_from_sgml(sgml_content: str) -> str:
    """
    Extract the primary 10-K document text from an SGML-wrapped filing.

    Uses secsgml to parse the SGML envelope and extract the primary HTML document,
    then converts it to clean text.
    """
    try:
        from secsgml import parse_sgml_submission  # type: ignore

        # Parse SGML into documents
        documents = parse_sgml_submission(sgml_content)

        # Find the primary 10-K document
        primary_doc = None
        for doc in documents:
            doc_type = doc.get("type", "").upper()
            if doc_type in ("10-K", "10-K/A"):
                primary_doc = doc
                break

        if primary_doc is None and documents:
            # Fallback: use the first document
            primary_doc = documents[0]

        if primary_doc is None:
            logger.warning("No documents found in SGML submission")
            return sgml_content  # Return raw content as fallback

        # Get the document content
        doc_content = primary_doc.get("text", primary_doc.get("content", ""))

        # Detect if the extracted content is HTML and convert
        if _detect_content_type(doc_content) == "html":
            return _html_to_text(doc_content)
        else:
            return _strip_html_tags(doc_content)

    except ImportError:
        logger.warning("secsgml not available, falling back to regex SGML extraction")
        return _extract_sgml_text_regex(sgml_content)
    except Exception as e:
        logger.warning("secsgml parsing failed: %s, falling back to regex", e)
        return _extract_sgml_text_regex(sgml_content)


def _extract_sgml_text_regex(sgml_content: str) -> str:
    """
    Simple regex-based SGML text extraction (fallback if secsgml fails).

    Extracts text between <TEXT> and </TEXT> tags from the primary 10-K document.
    """
    # Find all <DOCUMENT> blocks
    doc_pattern = re.compile(
        r"<DOCUMENT>\s*<TYPE>(.*?)\n.*?<TEXT>(.*?)</TEXT>",
        re.DOTALL | re.IGNORECASE,
    )

    for match in doc_pattern.finditer(sgml_content):
        doc_type = match.group(1).strip().upper()
        if doc_type in ("10-K", "10-K/A"):
            text_content = match.group(2)
            content_type = _detect_content_type(text_content)
            if content_type == "html":
                return _html_to_text(text_content)
            return _strip_html_tags(text_content)

    # If no 10-K document found, try to extract any <TEXT> block
    text_match = re.search(r"<TEXT>(.*?)</TEXT>", sgml_content, re.DOTALL | re.IGNORECASE)
    if text_match:
        return _strip_html_tags(text_match.group(1))

    # Ultimate fallback
    return _strip_html_tags(sgml_content)


# ---------------------------------------------------------------------------
#  Download via datamule (Layer 2 middle fallback)
# ---------------------------------------------------------------------------

def _download_via_sgml_path(cik: str, accession_number: str) -> str | None:
    """
    Download full submission .txt file and parse SGML if needed.

    This handles the SGML format issue that edgartools can't resolve.
    Downloads the raw submission text file (which may be SGML-wrapped)
    and uses secsgml (or regex fallback) to extract the primary document.

    Includes retry with exponential backoff for transient SEC errors.
    Returns cleaned text or None on failure.
    """
    retry_cfg = _get_retry_config()

    @retry(
        wait=wait_exponential(min=retry_cfg["backoff_min"], max=retry_cfg["backoff_max"]),
        stop=stop_after_attempt(retry_cfg["max_retries"]),
        retry=retry_if_exception_type(_RETRYABLE_HTTP_EXCEPTIONS) | retry_if_exception_type(httpx.HTTPStatusError),
        before_sleep=lambda rs: logger.warning(
            "SGML path retry %d/%d after %s — waiting %.0fs...",
            rs.attempt_number, retry_cfg["max_retries"],
            rs.outcome.exception().__class__.__name__,
            rs.next_action.sleep,
        ),
        reraise=True,
    )
    def _download():
        acc_clean = accession_number.replace("-", "")
        cik_clean = cik.lstrip("0") or "0"

        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_clean}/{acc_clean}/{accession_number}.txt"
        )

        email = _get_sec_email()
        time.sleep(_get_rate_limit())

        with httpx.Client(
            timeout=httpx.Timeout(float(_get_http_timeout()), connect=30.0),
            headers={"User-Agent": email, "Accept-Encoding": "gzip, deflate"},
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()

        content = resp.text
        content_type = _detect_content_type(content)

        if content_type == "sgml":
            logger.info("SGML path: detected SGML content, extracting primary doc...")
            return _extract_primary_doc_from_sgml(content)
        elif content_type == "html":
            logger.info("SGML path: detected HTML content, converting...")
            return _html_to_text(content)
        else:
            return content

    try:
        return _download()
    except Exception as e:
        logger.warning("SGML path download failed after retries: %s", e)
        return None


# ---------------------------------------------------------------------------
#  Direct SEC EDGAR download (Layer 2 last fallback)
# ---------------------------------------------------------------------------

# Retryable HTTP exceptions
_RETRYABLE_HTTP_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Check if an HTTP status error is retryable (503, 429)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503)
    return isinstance(exc, _RETRYABLE_HTTP_EXCEPTIONS)


def _direct_download_filing_text(
    cik: str, accession_number: str, email: str
) -> str:
    """
    Download 10-K filing text directly from SEC EDGAR.

    Bypasses edgartools' internal download which has timeout/format issues.
    Uses httpx with generous timeout and proper User-Agent.

    Has built-in retry via tenacity for transient HTTP errors (503, 429, timeouts).
    """
    retry_cfg = _get_retry_config()
    timeout = _get_http_timeout()
    rate_limit = _get_rate_limit()

    @retry(
        wait=wait_exponential(min=retry_cfg["backoff_min"], max=retry_cfg["backoff_max"]),
        stop=stop_after_attempt(retry_cfg["max_retries"]),
        retry=retry_if_exception_type(_RETRYABLE_HTTP_EXCEPTIONS) | retry_if_exception_type(httpx.HTTPStatusError),
        before_sleep=lambda rs: logger.warning(
            "Retry %d/%d after %s — waiting %.0fs...",
            rs.attempt_number, retry_cfg["max_retries"],
            rs.outcome.exception().__class__.__name__,
            rs.next_action.sleep,
        ),
        reraise=True,
    )
    def _download_with_retry() -> str:
        # Format accession number for URL (remove dashes)
        acc_no_dashes = accession_number.replace("-", "")
        cik_clean = cik.lstrip("0") or "0"

        # Step 1: Get the filing index page
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_clean}/{acc_no_dashes}/{accession_number}-index.htm"
        )

        logger.info("Direct download: fetching index %s", index_url)

        with httpx.Client(
            timeout=httpx.Timeout(float(timeout), connect=30.0),
            headers={"User-Agent": email, "Accept-Encoding": "gzip, deflate"},
            follow_redirects=True,
        ) as client:
            # Download the filing index page
            time.sleep(rate_limit)
            resp = client.get(index_url)
            resp.raise_for_status()

            # Step 2: Parse index to find the primary 10-K document
            primary_doc_url = _find_primary_document(resp.text, cik_clean, acc_no_dashes)

            if not primary_doc_url:
                # Fallback: try the full submission text file
                logger.warning("Could not find primary doc in index, trying full text file")
                txt_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_clean}/{acc_no_dashes}/{accession_number}.txt"
                )
                time.sleep(rate_limit)
                resp = client.get(txt_url)
                resp.raise_for_status()

                content_type = _detect_content_type(resp.text)
                if content_type == "sgml":
                    return _extract_primary_doc_from_sgml(resp.text)
                return _strip_html_tags(resp.text)

            # Step 3: Download the primary document
            logger.info("Direct download: fetching primary doc %s", primary_doc_url)
            time.sleep(rate_limit)
            resp = client.get(primary_doc_url)
            resp.raise_for_status()

            # Step 4: Convert based on content type
            content_type = _detect_content_type(resp.text)
            if content_type == "sgml":
                return _extract_primary_doc_from_sgml(resp.text)
            return _html_to_text(resp.text)

    return _download_with_retry()


def _find_primary_document(
    index_html: str, cik: str, acc_no_dashes: str
) -> str | None:
    """Parse the filing index page to find the primary 10-K HTML document URL."""
    try:
        tree = lxml_html.fromstring(index_html)

        # Look for the table with filing documents
        for row in tree.xpath("//table//tr"):
            cells = row.xpath("td")
            if len(cells) >= 4:
                doc_type = cells[3].text_content().strip()
                if doc_type in ("10-K", "10-K/A"):
                    link = cells[2].xpath(".//a/@href")
                    if link:
                        href = link[0]
                        if href.startswith("/"):
                            return f"https://www.sec.gov{href}"
                        return href
    except Exception as e:
        logger.warning("Could not parse filing index: %s", e)

    return None


# ---------------------------------------------------------------------------
#  HTML / text conversion
# ---------------------------------------------------------------------------

def _html_to_text(html_content: str) -> str:
    """Convert HTML to clean text using lxml, with improved table handling."""
    try:
        tree = lxml_html.fromstring(html_content)

        # Remove script and style elements
        for element in tree.xpath("//script | //style"):
            element.getparent().remove(element)

        text = tree.text_content()
    except Exception:
        # Fallback: simple regex-based HTML stripping
        text = _strip_html_tags(html_content)

    # Clean up whitespace
    text = re.sub(r"[ \t]+", " ", text)  # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse vertical whitespace
    text = text.strip()

    return text


def _strip_html_tags(html_content: str) -> str:
    """Simple regex-based HTML tag removal (ultimate fallback)."""
    text = re.sub(r"<[^>]+>", " ", html_content)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
#  Section extraction
# ---------------------------------------------------------------------------

def _extract_sections_from_text(full_text: str) -> dict[str, str]:
    """
    Extract 10-K sections from plain text using regex patterns.

    Used as fallback / gap-filler when edgartools' TenK parser returns
    incomplete sections.
    """
    sections = {}
    text_upper = full_text.upper()

    # Find all section starts
    section_positions = []
    for section_name, pattern in SECTION_PATTERNS.items():
        for match in re.finditer(pattern, text_upper):
            section_positions.append((match.start(), section_name))

    # Sort by position in document
    section_positions.sort(key=lambda x: x[0])

    # Extract content between section headers
    for i, (start_pos, section_name) in enumerate(section_positions):
        # Find end: next section start or end of document (max 500k chars)
        if i + 1 < len(section_positions):
            end_pos = section_positions[i + 1][0]
        else:
            end_pos = min(start_pos + 500_000, len(full_text))

        content = full_text[start_pos:end_pos].strip()

        # Remove the header line itself
        lines = content.split("\n", 1)
        if len(lines) > 1:
            content = lines[1].strip()

        if len(content) > 100:  # Only keep sections with meaningful content
            sections[section_name] = content
            logger.debug(
                "Regex-extracted %s: %d chars", section_name, len(content)
            )

    return sections


def _extract_sections_edgartools(filing) -> dict[str, str]:
    """Extract key sections using edgartools' TenK parser.

    Uses two complementary extraction paths:
    1. Direct property access (business, risk_factors, ...) for sections
       that expose a stable attribute on the TenK object.
    2. `TenK.__getitem__("Item N")` for sections without a direct property
       (currently Item 8 / Financial Statements). This routes through
       edgartools' modern `sections`-parser and is markedly more robust
       against atypical filings than the regex fallback.
    """
    sections = {}
    tenk = filing.obj()

    for attr_name, section_name in TENK_DIRECT_ATTRS.items():
        try:
            content = getattr(tenk, attr_name, None)
            if content is not None:
                text = str(content).strip()
                if text:
                    sections[section_name] = text
        except Exception as e:
            logger.warning("Could not extract .%s: %s", attr_name, e)

    for item_key, section_name in TENK_ITEM_LOOKUPS.items():
        try:
            content = tenk[item_key]
            if content is not None:
                text = str(content).strip()
                if text:
                    sections[section_name] = text
        except Exception as e:
            logger.warning("Could not extract %s via __getitem__: %s", item_key, e)

    return sections


def _extract_sections_hybrid(filing, full_text: str) -> dict[str, str]:
    """
    Hybrid section extraction: edgartools primary, regex fills gaps.

    Strategy:
    1. Try edgartools TenK parser for all available sections
    2. Run regex extraction on full text
    3. For any section keys that edgartools missed, fill from regex results
    """
    # Step 1: edgartools as primary source
    sections = {}
    try:
        logger.info("Extracting sections via edgartools TenK parser...")
        sections = _extract_sections_edgartools(filing)
        logger.info("edgartools extracted %d sections: %s", len(sections), list(sections.keys()))
    except Exception as e:
        logger.warning("TenK parser failed: %s", e)

    # Step 2: regex extraction
    logger.info("Running regex section extraction to fill gaps...")
    regex_sections = _extract_sections_from_text(full_text)
    logger.info("Regex found %d sections: %s", len(regex_sections), list(regex_sections.keys()))

    # Step 3: fill gaps — only add sections that edgartools didn't find
    for section_name, content in regex_sections.items():
        if section_name not in sections:
            sections[section_name] = content
            logger.info("Regex filled gap: %s (%d chars)", section_name, len(content))

    if sections:
        logger.info("Total: %d sections extracted", len(sections))
    else:
        logger.warning("No sections extracted. Full text will be used as fallback.")

    return sections


# ---------------------------------------------------------------------------
#  Main download function
# ---------------------------------------------------------------------------


def _find_existing_filing(
    ticker: str, fiscal_year: int | None = None
) -> ProcessedFiling | None:
    """
    Check if a processed filing already exists on disk.

    For a specific fiscal_year, matches files where the fiscal_year_end
    in the metadata starts with the requested year.
    For fiscal_year=None (latest), returns the most recent processed filing.

    Returns ProcessedFiling if found, None otherwise.
    """
    ticker_dir = DATA_PROCESSED / ticker.upper()

    if not ticker_dir.exists():
        return None

    meta_files = sorted(ticker_dir.glob("10K_*.meta.json"), reverse=True)
    if not meta_files:
        return None

    for meta_path in meta_files:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            if fiscal_year is not None:
                # Match on fiscal_year_end field
                fye = meta.get("fiscal_year_end", "")
                if not fye.startswith(str(fiscal_year)):
                    continue

            # Found a match — load the full filing
            md_path = meta_path.with_suffix("").with_suffix(".md")
            if not md_path.exists():
                continue

            filing = ProcessedFiling.from_files(md_path, meta_path)
            logger.info(
                "⏭ Skipping download — %s FY%s already exists: %s",
                ticker, fiscal_year or "latest", md_path.name,
            )
            return filing

        except Exception as e:
            logger.debug("Could not load existing filing %s: %s", meta_path, e)
            continue

    return None

def download_filing(
    ticker: str, fiscal_year: int | None = None, *, force_download: bool = False
) -> ProcessedFiling:
    """
    Download a 10-K filing for a given ticker.

    Checks if processed files already exist and skips download if so.
    Use force_download=True to re-download regardless.

    Three-layer strategy:
      Layer 1 — Discovery:  edgartools finds the right filing
      Layer 2 — Download:   edgartools → SGML path → direct httpx
      Layer 3 — Parse:      edgartools TenK → regex fills gaps → full text

    All download attempts use tenacity retry with exponential backoff
    for transient errors (HTTP 503, 429, timeouts).

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        fiscal_year: Optional fiscal year (e.g., 2024). If None, downloads the latest.
        force_download: If True, re-download even if files already exist.

    Returns:
        ProcessedFiling with metadata, extracted sections, and full text.
    """
    # --- Check for existing processed files ---
    if not force_download:
        existing = _find_existing_filing(ticker, fiscal_year)
        if existing is not None:
            return existing

    _init_edgar()
    retry_cfg = _get_retry_config()

    logger.info("Fetching 10-K filing for %s (FY%s)...", ticker, fiscal_year or "latest")

    # --- Layer 1: Discovery (with retry for transient network errors) ---
    @retry(
        wait=wait_exponential(min=retry_cfg["backoff_min"], max=retry_cfg["backoff_max"]),
        stop=stop_after_attempt(retry_cfg["max_retries"]),
        retry=retry_if_exception_type(_RETRYABLE_HTTP_EXCEPTIONS) | retry_if_exception_type(httpx.HTTPStatusError),
        before_sleep=lambda rs: logger.warning(
            "Discovery retry %d/%d after %s — waiting %.0fs...",
            rs.attempt_number, retry_cfg["max_retries"],
            rs.outcome.exception().__class__.__name__,
            rs.next_action.sleep,
        ),
        reraise=True,
    )
    def _discover_filing():
        company = Company(ticker)
        if fiscal_year:
            target = _find_filing_by_fiscal_year(company, fiscal_year)
        else:
            filings = company.get_filings(form="10-K")
            target = filings.latest()
            if target is None:
                raise ValueError(f"No 10-K filing found for {ticker}")
        return company, target

    company, target = _discover_filing()

    logger.info(
        "Found 10-K filing for %s filed on %s (period: %s)",
        ticker,
        target.filing_date,
        getattr(target, "period_of_report", "unknown"),
    )

    # --- Extract metadata ---
    metadata = FilingMetadata(
        ticker=ticker.upper(),
        company_name=company.name,
        cik=str(company.cik),
        filing_date=str(target.filing_date),
        accession_number=str(target.accession_number),
        fiscal_year_end=str(getattr(target, "period_of_report", "unknown")),
    )

    # --- Layer 2: Download (cascade with retry) ---
    full_text = None

    # Attempt 1: edgartools
    try:
        logger.info("[1/3] Trying edgartools download...")
        full_text = target.text()
        if full_text and len(full_text) > 500:
            logger.info("edgartools download succeeded: %d chars", len(full_text))
        else:
            logger.warning("edgartools returned insufficient text (%d chars)", len(full_text or ""))
            full_text = None
    except Exception as e:
        logger.warning("edgartools download failed: %s", e)

    # Attempt 2: SGML path (handles SGML format directly)
    if not full_text:
        logger.info("[2/3] Trying SGML path (full submission .txt)...")
        full_text = _download_via_sgml_path(
            cik=metadata.cik,
            accession_number=metadata.accession_number,
        )
        if full_text:
            logger.info("SGML path download succeeded: %d chars", len(full_text))

    # Attempt 3: direct httpx download
    if not full_text:
        logger.info("[3/3] Falling back to direct SEC EDGAR download...")
        email = _get_sec_email()
        full_text = _direct_download_filing_text(
            cik=metadata.cik,
            accession_number=metadata.accession_number,
            email=email,
        )
        logger.info("Direct download succeeded: %d chars", len(full_text))

    # --- Layer 3: Parse (hybrid section extraction) ---
    sections = _extract_sections_hybrid(target, full_text)

    # --- Save ---
    # Save raw text
    raw_dir = DATA_RAW / ticker.upper()
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"10K_{metadata.filing_date}.txt"
    raw_file.write_text(full_text, encoding="utf-8")
    logger.info("Raw text saved to %s", raw_file)

    filing = ProcessedFiling(
        metadata=metadata,
        sections=sections,
        full_text=full_text,
    )

    # Save processed output
    _save_processed(filing)

    return filing


def _find_filing_by_fiscal_year(company: Company, fiscal_year: int):
    """Find a 10-K filing for a specific fiscal year."""
    # Filter by filing_date year range to avoid iterating ALL filings
    search_filings = company.get_filings(
        form="10-K",
        date=f"{fiscal_year}-01-01:{fiscal_year + 1}-12-31",
    )
    for f in search_filings:
        period = str(getattr(f, "period_of_report", ""))
        if period.startswith(str(fiscal_year)):
            return f

    raise ValueError(
        f"No 10-K filing found for {company.name} FY{fiscal_year}. "
        f"Searched filings from {fiscal_year}-01 to {fiscal_year + 1}-12."
    )


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------

def _save_processed(filing: ProcessedFiling) -> Path:
    """
    Save processed filing as Markdown + JSON metadata sidecar.

    Output files:
      data/processed/{TICKER}/10K_{date}.md         — Structured Markdown
      data/processed/{TICKER}/10K_{date}.meta.json   — Machine-readable metadata
    """
    ticker = filing.metadata.ticker
    date = filing.metadata.filing_date

    out_dir = DATA_PROCESSED / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save Markdown
    md_file = out_dir / f"10K_{date}.md"
    md_file.write_text(filing.to_markdown(), encoding="utf-8")
    logger.info("Markdown saved to %s", md_file)

    # Save metadata sidecar
    meta_file = out_dir / f"10K_{date}.meta.json"
    meta_file.write_text(
        json.dumps(filing.to_metadata_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Metadata saved to %s", meta_file)

    return md_file


def _parse_markdown_sections(markdown_text: str) -> dict[str, str]:
    """Parse a structured Markdown file back into sections by ## headers."""
    sections = {}
    current_section = None
    current_lines: list[str] = []

    for line in markdown_text.split("\n"):
        if line.startswith("## "):
            # Save previous section
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = line[3:].strip()
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    # Save last section
    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def load_processed_filing(
    ticker: str, filing_date: str | None = None
) -> ProcessedFiling:
    """
    Load a previously processed 10-K filing from disk.

    Args:
        ticker: Stock ticker symbol.
        filing_date: Specific filing date. If None, loads the latest.

    Returns:
        ProcessedFiling object.
    """
    ticker_dir = DATA_PROCESSED / ticker.upper()

    if not ticker_dir.exists():
        raise FileNotFoundError(
            f"No processed filings found for {ticker}. "
            "Run download_filing() first."
        )

    if filing_date:
        md_path = ticker_dir / f"10K_{filing_date}.md"
        meta_path = ticker_dir / f"10K_{filing_date}.meta.json"
    else:
        # Get latest by filename sort
        files = sorted(ticker_dir.glob("10K_*.md"), reverse=True)
        if not files:
            raise FileNotFoundError(f"No processed filings in {ticker_dir}")
        md_path = files[0]
        meta_path = md_path.with_suffix("").with_suffix(".meta.json")

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata sidecar not found: {meta_path}")

    return ProcessedFiling.from_files(md_path, meta_path)


def download_all_filings() -> list[ProcessedFiling]:
    """
    Download 10-K filings for all configured companies.

    Companies are defined in configs/base.yaml.
    """
    config = load_config()
    companies = config.get("companies", [])

    results = []
    for company_info in companies:
        ticker = company_info["ticker"]
        logger.info("=" * 50)
        logger.info("Processing %s (%s)", company_info["name"], ticker)
        logger.info("=" * 50)
        try:
            filing = download_filing(ticker)
            results.append(filing)
            logger.info(
                "✓ %s: %d sections extracted, %d chars total",
                ticker,
                len(filing.sections),
                len(filing.full_text),
            )
        except Exception as e:
            logger.error("✗ Failed to process %s: %s", ticker, e)

    return results


# --- CLI Entry Point ---
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) > 1:
        # Download specific ticker
        ticker_arg = sys.argv[1]
        download_filing(ticker_arg)
    else:
        # Download all configured companies
        download_all_filings()
