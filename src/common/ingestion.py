"""
SEC EDGAR 10-K Filing Ingestion Pipeline.

Downloads 10-K filings from SEC EDGAR using edgartools,
extracts key sections, and saves as Markdown for downstream processing.

Output format: Markdown (.md) + metadata sidecar (.meta.json)
- Markdown: Structured text with headers for semantic chunking (RAG)
  and token-efficient full-context input (Long Context).
- JSON sidecar: Machine-readable metadata (ticker, CIK, dates).

Shared by ALL 4 systems for fair comparison.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from edgar import Company, set_identity

from src.common.config import load_config

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


# Direct attributes on the TenK object (most reliable)
TENK_DIRECT_ATTRS = {
    "business": "Business",
    "risk_factors": "Risk Factors",
    "management_discussion": "MD&A",
    "directors_officers_and_governance": "Directors and Corporate Governance",
}


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


def _init_edgar() -> None:
    """Initialize edgartools with SEC-required identity."""
    config = load_config()
    email = config.get("sec_edgar_email")
    if not email:
        raise ValueError(
            "SEC_EDGAR_EMAIL not set. SEC requires identification. "
            "Set it in your .env file."
        )
    set_identity(email)
    logger.info("EDGAR identity set to: %s", email)


def download_filing(ticker: str) -> ProcessedFiling:
    """
    Download the latest 10-K filing for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').

    Returns:
        ProcessedFiling with metadata, extracted sections, and full text.
    """
    _init_edgar()

    logger.info("Fetching 10-K filing for %s...", ticker)
    company = Company(ticker)
    filings = company.get_filings(form="10-K")
    latest = filings.latest()

    if latest is None:
        raise ValueError(f"No 10-K filing found for {ticker}")

    logger.info(
        "Found 10-K filing for %s filed on %s",
        ticker,
        latest.filing_date,
    )

    # Extract metadata
    metadata = FilingMetadata(
        ticker=ticker.upper(),
        company_name=company.name,
        cik=str(company.cik),
        filing_date=str(latest.filing_date),
        accession_number=str(latest.accession_number),
        fiscal_year_end=str(getattr(latest, "period_of_report", "unknown")),
    )

    # Get full text (clean, readable)
    logger.info("Extracting full text...")
    full_text = latest.text()

    # Save raw text
    raw_dir = DATA_RAW / ticker.upper()
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"10K_{metadata.filing_date}.txt"
    raw_file.write_text(full_text, encoding="utf-8")
    logger.info("Raw text saved to %s", raw_file)

    # Extract sections from the TenK object
    logger.info("Extracting 10-K sections...")
    sections = _extract_sections(latest)

    filing = ProcessedFiling(
        metadata=metadata,
        sections=sections,
        full_text=full_text,
    )

    # Save processed output
    _save_processed(filing)

    return filing


def _extract_sections(filing) -> dict[str, str]:
    """
    Extract key sections from a 10-K filing.

    Strategy:
    1. Try direct named attributes (business, risk_factors, etc.)
    2. Fall back to the sections object for remaining items
    """
    sections = {}

    try:
        tenk = filing.obj()

        # 1. Extract via direct named attributes (most reliable)
        for attr_name, section_name in TENK_DIRECT_ATTRS.items():
            try:
                content = getattr(tenk, attr_name, None)
                if content is not None:
                    text = str(content).strip()
                    if text:
                        sections[section_name] = text
                        logger.debug(
                            "Extracted %s via .%s: %d chars",
                            section_name,
                            attr_name,
                            len(text),
                        )
            except Exception as e:
                logger.warning("Could not extract .%s: %s", attr_name, e)

        # 2. Try the sections object for additional items
        try:
            tenk_sections = tenk.sections
            if tenk_sections:
                for item_name in tenk.items:
                    # Skip items we already extracted via direct attributes
                    if any(item_name.lower().replace(" ", "") in key.lower()
                           for key in sections):
                        continue
                    try:
                        section = tenk.get_item_with_part(item_name)
                        if section:
                            text = str(section).strip()
                            if text:
                                sections[item_name] = text
                                logger.debug(
                                    "Extracted %s via sections: %d chars",
                                    item_name,
                                    len(text),
                                )
                    except Exception as e:
                        logger.debug("Could not extract %s: %s", item_name, e)
        except Exception as e:
            logger.debug("Could not iterate sections: %s", e)

    except Exception as e:
        logger.warning(
            "Could not parse TenK object, using full text only: %s", e
        )

    if not sections:
        logger.warning(
            "No sections extracted via TenK object. "
            "Full text is still available for processing."
        )
    else:
        logger.info("Extracted %d sections from TenK object", len(sections))

    return sections


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


def load_processed_filing(ticker: str, filing_date: str | None = None) -> ProcessedFiling:
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
