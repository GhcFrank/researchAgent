"""Official SEC EDGAR retrieval only; no extraction, persistence, or caching."""

from email.message import Message
from datetime import date
import gzip
from html.parser import HTMLParser
from http.client import HTTPException
import json
import os
import re
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.parse import quote, unquote
from urllib.request import Request, build_opener
import zlib

from research_tools import ResearchMaterial, ResearchTool, ResearchToolError, ResearchToolSpec, SearchResult


class SECResearchToolError(ResearchToolError):
    """Base error for SEC retrieval or invalid tool arguments."""


class SECConfigurationError(SECResearchToolError):
    """SEC_USER_AGENT is not configured for scripted access."""


class SECTickerNotFoundError(SECResearchToolError):
    """No exact ticker match exists in the official SEC mapping."""


class SECResponseError(SECResearchToolError):
    """An HTTP response or SEC filing metadata cannot be used."""


_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/"
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/"
_FORMS = ("10-K", "10-Q", "8-K")
_RECENT_FIELDS = {
    "form": "form",
    "filing_date": "filingDate",
    "accession_number": "accessionNumber",
    "primary_document": "primaryDocument",
}
_OPTIONAL_FIELDS = {"report_date": "reportDate", "primary_doc_description": "primaryDocDescription"}


class _ReadableHTMLParser(HTMLParser):
    """Preserve text blocks and table rows without interpreting SEC sections."""

    _BLOCK_TAGS = {"p", "div", "section", "article", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"}
    _VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    _INLINE_NAMESPACES = {"http://www.xbrl.org/2008/inlineXBRL", "http://www.xbrl.org/2013/inlineXBRL"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.parts = []
        self.hidden_stack = []
        self.inline_prefixes = {"ix"}
        self.table_depth = 0

    def _is_hidden(self, tag, attrs):
        # HTMLParser supplies colon-prefixed names, lowercased by the parser.
        # Also accept declared Inline XBRL aliases instead of hardcoding one prefix.
        for name, value in attrs:
            name = name.casefold()
            if name.startswith("xmlns:") and value in self._INLINE_NAMESPACES:
                self.inline_prefixes.add(name.split(":", 1)[1])
        prefix, _, local = tag.partition(":")
        if tag.rsplit(":", 1)[-1] in ("script", "style") or (
            prefix in self.inline_prefixes and local in ("header", "hidden")
        ):
            return True
        # Inspect only inline display declarations, not other styles or CSS rules.
        display, important = None, False
        for name, value in attrs:
            if name.casefold() != "style" or not value:
                continue
            for declaration in value.split(";"):
                property_name, separator, setting = declaration.partition(":")
                if separator and property_name.strip().casefold() == "display":
                    setting, _, priority = setting.partition("!")
                    new_important = priority.strip().casefold() == "important"
                    if new_important or not important:
                        display, important = setting.strip().casefold(), new_important
        return display == "none"

    def flush_block(self):
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        text = "\n".join(line for line in lines if line)
        if text:
            self.blocks.append(text)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        tag = tag.casefold()
        local = tag.rsplit(":", 1)[-1]
        if self.hidden_stack or self._is_hidden(tag, attrs):
            # Void elements have no closing tag; they must not keep a subtree hidden.
            if local not in self._VOID_TAGS:
                self.hidden_stack.append(tag)
            return
        if local == "table":
            if self.table_depth == 0:
                self.flush_block()
            self.table_depth += 1
        elif local == "br":
            self.parts.append("\n")
        elif local in self._BLOCK_TAGS:
            if self.table_depth:
                self.parts.append(" ")
            else:
                self.flush_block()

    def handle_endtag(self, tag):
        tag = tag.casefold()
        local = tag.rsplit(":", 1)[-1]
        if self.hidden_stack:
            # Closing a parent also closes any unclosed descendants within it.
            for index in range(len(self.hidden_stack) - 1, -1, -1):
                if self.hidden_stack[index] == tag:
                    del self.hidden_stack[index:]
                    break
            return
        if local == "table" and self.table_depth:
            self.table_depth -= 1
            if self.table_depth == 0:
                self.flush_block()
        elif self.table_depth:
            if local == "tr":
                self.parts.append("\n")
            elif local in ("td", "th"):
                self.parts.append("\t")
            elif local in self._BLOCK_TAGS:
                self.parts.append(" ")
        elif local in self._BLOCK_TAGS:
            self.flush_block()

    def handle_data(self, data):
        if not self.hidden_stack:
            # Source indentation/newlines are HTML whitespace, not block breaks.
            # Keep only the structural newlines introduced for br/table rows.
            self.parts.append(re.sub(r"\s+", " ", data))


def html_to_readable_text(content: str) -> str:
    """Remove hidden subtrees; retain visible inline facts and readable blocks.

    Only inline display:none is interpreted; external stylesheets and computed
    CSS are outside this helper. Visible Inline XBRL values are kept as displayed,
    without interpreting scale, sign, taxonomy, or other XBRL attributes.
    """
    parser = _ReadableHTMLParser()
    parser.feed(content)
    parser.close()
    parser.flush_block()
    return "\n\n".join(parser.blocks)


def _cik(value) -> str:
    digits = str(value) if type(value) is int else value
    if not isinstance(digits, str) or not re.fullmatch(r"[0-9]{1,10}", digits) or int(digits) == 0:
        raise SECResponseError("Invalid SEC CIK: expected a positive number of at most ten digits")
    return digits.zfill(10)


def _text(record: dict, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SECResponseError(f"Missing or invalid SEC metadata: {field}")
    return value.strip()


def _filing_identity(cik: str, accession: str, document: str) -> tuple[str, str]:
    if not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession):
        raise SECResponseError("Invalid SEC accession_number")
    if any(part in ("", ".", "..") for part in document.split("/")) or "\\" in document:
        raise SECResponseError("Invalid SEC primary_document path")
    source_ref = f"sec:{cik}:{accession}:{quote(document, safe='')}"
    locator = f"{_ARCHIVES_URL}{int(cik)}/{accession.replace('-', '')}/{quote(document, safe='/')}"
    return source_ref, locator


class SECResearchTool(ResearchTool):
    """search accepts a ticker; read resolves a stable reference without a cache.

    client may be a urllib-compatible opener implementing open(request, timeout),
    with context-managed responses exposing status, headers and read(). Default
    requests are synchronous, have a 30-second timeout and do not retry. A single
    instance spaces request starts by at least 0.11 seconds; it is not thread-safe.
    Callers load .env themselves; only SEC_USER_AGENT is read here.
    """

    spec = ResearchToolSpec(
        name="sec",
        description=(
            "Search and retrieve official SEC EDGAR filings for a public company. "
            "Use this tool when primary SEC filings such as 10-K, 10-Q, or 8-K are needed."
        ),
        capabilities=("search", "read"),
        source_types=_FORMS,
    )

    def __init__(self, client=None):
        self.user_agent = os.getenv("SEC_USER_AGENT", "").strip()
        if not self.user_agent or "\r" in self.user_agent or "\n" in self.user_agent:
            raise SECConfigurationError("SEC_USER_AGENT must be configured as a non-blank single line")
        self.client = client if client is not None else build_opener()
        self._last_request_at = None

    def _request(self, url: str) -> tuple[bytes, Message]:
        if self._last_request_at is not None:
            delay = self._last_request_at + 0.11 - monotonic()
            if delay > 0:
                sleep(delay)
        self._last_request_at = monotonic()
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"})
        try:
            with self.client.open(request, timeout=30.0) as response:
                if response.status >= 400:
                    raise SECResponseError(f"SEC HTTP {response.status} for {url}")
                headers, body = response.headers, response.read()
            encoding = headers.get("Content-Encoding", "").strip().lower()
            if encoding == "gzip":
                body = gzip.decompress(body)
            elif encoding == "deflate":
                try:
                    body = zlib.decompress(body)
                except zlib.error:
                    body = zlib.decompress(body, -zlib.MAX_WBITS)
            elif encoding not in ("", "identity"):
                raise SECResponseError(f"Unsupported SEC content encoding: {encoding}")
            return body, headers
        except HTTPError as exc:
            raise SECResponseError(f"SEC HTTP {exc.code} for {url}") from exc
        except (OSError, HTTPException, ValueError, EOFError, zlib.error) as exc:
            raise SECResponseError(f"SEC request failed for {url}: {type(exc).__name__}") from exc

    def _json(self, url: str) -> dict:
        body, _ = self._request(url)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SECResponseError(f"Invalid SEC JSON at {url}") from exc
        if not isinstance(payload, dict):
            raise SECResponseError(f"SEC JSON at {url} must be an object")
        return payload

    def _resolve_company(self, ticker: str) -> dict:
        company = None
        for record in self._json(_TICKERS_URL).values():
            if not isinstance(record, dict):
                raise SECResponseError("Invalid SEC ticker map record")
            name, symbol = _text(record, "title"), _text(record, "ticker").upper()
            cik = _cik(record.get("cik_str"))
            if symbol == ticker:
                company = {"cik": cik, "company_name": name, "ticker": symbol}
        if company is None:
            raise SECTickerNotFoundError(f"SEC ticker not found: {ticker}")
        return company

    def _recent_filings(self, cik: str, fallback_ticker: str | None = None) -> list[SearchResult]:
        payload = self._json(f"{_SUBMISSIONS_URL}CIK{cik}.json")
        company_name = _text(payload, "name")
        if "cik" in payload and _cik(payload["cik"]) != cik:
            raise SECResponseError("SEC submissions CIK does not match the requested company")
        tickers = payload.get("tickers", [])
        if not isinstance(tickers, list) or any(not isinstance(ticker, str) or not ticker.strip() for ticker in tickers):
            raise SECResponseError("Invalid SEC submissions tickers")
        ticker = tickers[0].strip().upper() if tickers else fallback_ticker
        filings = payload.get("filings")
        recent = filings.get("recent") if isinstance(filings, dict) else None
        if not isinstance(recent, dict) or any(not isinstance(recent.get(field), list) for field in _RECENT_FIELDS.values()):
            raise SECResponseError("Missing or invalid SEC filings.recent arrays")
        size = len(recent["form"])
        columns = {**_RECENT_FIELDS, **_OPTIONAL_FIELDS}
        for field in columns.values():
            if field not in recent and field in _OPTIONAL_FIELDS.values():
                continue
            if not isinstance(recent.get(field), list) or len(recent[field]) != size:
                raise SECResponseError("SEC filings.recent arrays have inconsistent lengths")
        records = []
        for index, form in enumerate(recent["form"]):
            if not isinstance(form, str) or not form.strip():
                raise SECResponseError("Invalid SEC filing form")
            if form not in _FORMS:
                continue
            metadata = {name: recent[field][index] if field in recent else None for name, field in columns.items()}
            filing_date = _text(metadata, "filing_date")
            try:
                if date.fromisoformat(filing_date).isoformat() != filing_date:
                    raise ValueError("non-canonical date")
            except ValueError as exc:
                raise SECResponseError("Invalid SEC filing_date") from exc
            accession = _text(metadata, "accession_number")
            document = _text(metadata, "primary_document")
            metadata.update(filing_date=filing_date, accession_number=accession, primary_document=document)
            for field in _OPTIONAL_FIELDS:
                if metadata[field] is not None and not isinstance(metadata[field], str):
                    raise SECResponseError(f"Invalid SEC filing metadata: {field}")
            source_ref, locator = _filing_identity(cik, accession, document)
            records.append({
                **metadata, "cik": cik, "company_name": company_name, "ticker": ticker,
                "source_ref": source_ref, "title": f"{company_name} {form} {filing_date}",
                "source_type": form, "locator": locator,
                "publisher": "U.S. Securities and Exchange Commission", "published_date": filing_date,
                "primary_or_secondary": "Primary", "independence_group": f"sec:issuer:{cik}",
            })
        return sorted(records, key=lambda record: record["filing_date"], reverse=True)

    def _search(self, query: str) -> list[SearchResult]:
        if not isinstance(query, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", query.strip()):
            raise SECResearchToolError("SEC search query must be a public-company ticker, not natural language")
        company = self._resolve_company(query.strip().upper())
        return self._recent_filings(company["cik"], company["ticker"])[:20]

    def _read(self, source_ref: str) -> ResearchMaterial:
        if not isinstance(source_ref, str):
            raise SECResearchToolError("SEC source_ref must be a string")
        parts = source_ref.split(":", 3)
        if len(parts) != 4 or parts[0] != "sec" or not re.fullmatch(r"[0-9]{10}", parts[1]):
            raise SECResearchToolError("Invalid SEC source_ref")
        cik, accession = _cik(parts[1]), parts[2]
        try:
            document = unquote(parts[3], errors="strict")
        except UnicodeDecodeError as exc:
            raise SECResearchToolError("Invalid SEC source_ref encoding") from exc
        canonical_ref, _ = _filing_identity(cik, accession, document)
        if source_ref != canonical_ref:
            raise SECResearchToolError("SEC source_ref must use canonical encoding")
        filing = next((item for item in self._recent_filings(cik) if item["source_ref"] == source_ref), None)
        if filing is None:
            raise SECResponseError("SEC source_ref does not identify a supported recent filing")
        body, headers = self._request(filing["locator"])
        try:
            raw_content = body.decode(headers.get_content_charset() or "utf-8")
        except (UnicodeDecodeError, LookupError) as exc:
            raise SECResponseError("Cannot decode SEC primary document") from exc
        is_html = headers.get_content_type() in ("text/html", "application/xhtml+xml") or document.lower().endswith((".htm", ".html"))
        content = html_to_readable_text(raw_content) if is_html else raw_content
        if not content.strip():
            raise SECResponseError("Empty SEC primary document content")
        return {**filing, "content": content, "raw_content": raw_content, "content_type": headers.get("Content-Type", "")}
