"""
Unit tests for the SEC EDGAR ingestion pipeline.

Tests content type detection, SGML extraction, hybrid section extraction,
and the retry-decorated download helpers.
"""


from src.common.ingestion import (
    _detect_content_type,
    _extract_sections_from_text,
    _extract_sgml_text_regex,
    _html_to_text,
    _strip_html_tags,
)

# ---------------------------------------------------------------------------
#  Content type detection
# ---------------------------------------------------------------------------

class TestDetectContentType:
    """Tests for _detect_content_type()."""

    def test_html_with_doctype(self):
        content = "<!DOCTYPE html><html><body>Hello</body></html>"
        assert _detect_content_type(content) == "html"

    def test_html_with_table(self):
        content = "<TABLE><TR><TD>Cell</TD></TR></TABLE>"
        assert _detect_content_type(content) == "html"

    def test_sgml_with_document_tag(self):
        content = "<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>1\n<TEXT>Hello</TEXT>\n</DOCUMENT>"
        assert _detect_content_type(content) == "sgml"

    def test_sgml_with_sec_document(self):
        content = "<SEC-DOCUMENT>Some content here</SEC-DOCUMENT>"
        assert _detect_content_type(content) == "sgml"

    def test_plain_text(self):
        content = "This is just plain text without any markup."
        assert _detect_content_type(content) == "text"

    def test_empty_string(self):
        content = ""
        assert _detect_content_type(content) == "text"


# ---------------------------------------------------------------------------
#  SGML extraction (regex fallback)
# ---------------------------------------------------------------------------

class TestExtractSgmlTextRegex:
    """Tests for _extract_sgml_text_regex()."""

    def test_extracts_10k_text(self):
        sgml = (
            "<DOCUMENT>\n"
            "<TYPE>10-K\n"
            "<SEQUENCE>1\n"
            "<TEXT>\n"
            "This is the 10-K filing content.\n"
            "</TEXT>\n"
            "</DOCUMENT>\n"
            "<DOCUMENT>\n"
            "<TYPE>EX-31\n"
            "<SEQUENCE>2\n"
            "<TEXT>This is an exhibit.</TEXT>\n"
            "</DOCUMENT>"
        )
        result = _extract_sgml_text_regex(sgml)
        assert "10-K filing content" in result
        assert "exhibit" not in result

    def test_handles_html_inside_sgml(self):
        sgml = (
            "<DOCUMENT>\n"
            "<TYPE>10-K\n"
            "<TEXT>\n"
            "<html><body><p>Filing text</p></body></html>\n"
            "</TEXT>\n"
            "</DOCUMENT>"
        )
        result = _extract_sgml_text_regex(sgml)
        assert "Filing text" in result

    def test_fallback_to_any_text_block(self):
        sgml = (
            "<DOCUMENT>\n"
            "<TYPE>EX-31\n"
            "<TEXT>Some exhibit text</TEXT>\n"
            "</DOCUMENT>"
        )
        # No 10-K document, should fallback to first <TEXT> block
        result = _extract_sgml_text_regex(sgml)
        assert "exhibit text" in result


# ---------------------------------------------------------------------------
#  HTML to text conversion
# ---------------------------------------------------------------------------

class TestHtmlToText:
    """Tests for _html_to_text() and _strip_html_tags()."""

    def test_basic_html(self):
        html = "<html><body><p>Hello World</p></body></html>"
        result = _html_to_text(html)
        assert "Hello World" in result

    def test_removes_scripts(self):
        html = "<html><body><script>var x=1;</script><p>Text</p></body></html>"
        result = _html_to_text(html)
        assert "var x=1" not in result
        assert "Text" in result

    def test_collapses_whitespace(self):
        html = "<html><body><p>Hello    World</p></body></html>"
        result = _html_to_text(html)
        assert "Hello World" in result

    def test_strip_tags_handles_entities(self):
        html = "Price &amp; Value &lt;100&gt; with&nbsp;spaces"
        result = _strip_html_tags(html)
        assert "Price & Value" in result
        assert "<100>" in result


# ---------------------------------------------------------------------------
#  Section extraction (regex)
# ---------------------------------------------------------------------------

class TestExtractSectionsFromText:
    """Tests for _extract_sections_from_text()."""

    def test_extracts_business_section(self):
        text = (
            "Some preamble text\n\n"
            "ITEM 1. BUSINESS\n\n"
            "The Company designs and manufactures products. "
            "This section has enough content to exceed the 100 character minimum "
            "threshold that is required by the extraction function.\n\n"
            "ITEM 1A. RISK FACTORS\n\n"
            "The Company faces risks. "
            "This section also has enough content to exceed the 100 character minimum "
            "threshold required for extraction.\n\n"
        )
        sections = _extract_sections_from_text(text)
        assert "Business" in sections
        assert "Risk Factors" in sections

    def test_extracts_mda_section(self):
        text = (
            "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n\n"
            "Revenue increased 10% year over year. "
            "This provides a detailed discussion and analysis of the company's "
            "financial performance over the reporting period.\n\n"
        )
        sections = _extract_sections_from_text(text)
        assert "MD&A" in sections

    def test_skips_short_sections(self):
        text = (
            "ITEM 1. BUSINESS\n\n"
            "Too short.\n\n"
            "ITEM 1A. RISK FACTORS\n\n"
            "Also short.\n\n"
        )
        sections = _extract_sections_from_text(text)
        # Both sections are < 100 chars, should be skipped
        assert len(sections) == 0

    def test_empty_text(self):
        sections = _extract_sections_from_text("")
        assert sections == {}
