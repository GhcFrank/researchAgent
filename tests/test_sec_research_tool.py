"""Fake HTTP tests of SEC retrieval, stable references and readable content."""

from copy import deepcopy
from email.message import Message
import gzip
import json
from urllib.error import HTTPError, URLError
import zlib

import pytest

import sec_research_tool as sec_module
from research_tools import MockResearchTool, ResearchTool, ResearchToolSpec
from sec_research_tool import (
    SECConfigurationError,
    SECResearchTool,
    SECResearchToolError,
    SECResponseError,
    SECTickerNotFoundError,
    html_to_readable_text,
)
from tool_registry import ResearchToolRegistry


TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0001234567.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/report.htm"
SOURCE_REF = "sec:0001234567:0001234567-26-000001:report.htm"
USER_AGENT = "Local test suite tests@example.invalid"
TICKER_MAP = {"0": {"ticker": "EXM", "title": "Example Company", "cik_str": 1234567}}


class FakeResponse:
    def __init__(self, body, content_type="application/json", encoding=None, status=200):
        self.body = body if isinstance(body, bytes) else body.encode("utf-8")
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if encoding:
            self.headers["Content-Encoding"] = encoding

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeHTTP:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout, sec_module.monotonic()))
        assert request.full_url in self.routes, f"Unexpected HTTP request: {request.full_url}"
        response = self.routes[request.full_url]
        if isinstance(response, Exception):
            raise response
        return response


def row(number=1, form="10-Q", filing_date="2026-10-01", document="report.htm"):
    return {
        "form": form, "filingDate": filing_date, "reportDate": "2026-06-30",
        "accessionNumber": f"0001234567-26-{number:06d}",
        "primaryDocument": document, "primaryDocDescription": "Primary filing document",
    }


def submissions(rows):
    fields = tuple(row())
    return {
        "cik": 1234567, "name": "Example Company", "tickers": ["EXM"],
        "filings": {
            "recent": {field: [item[field] for item in rows] for field in fields},
            "files": [{"name": "older-submissions.json"}],
        },
    }


def json_response(payload):
    return FakeResponse(json.dumps(payload))


@pytest.fixture(autouse=True)
def fake_environment_and_clock(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", USER_AGENT)
    now = [0.0]
    monkeypatch.setattr(sec_module, "monotonic", lambda: now[0])
    monkeypatch.setattr(sec_module, "sleep", lambda duration: now.__setitem__(0, now[0] + duration))

    def forbidden_opener():
        raise AssertionError("All SEC tests must inject fake HTTP")

    monkeypatch.setattr(sec_module, "build_opener", forbidden_opener)


def test_search_metadata_case_insensitive_stable_refs_and_registry():
    records = [
        row(), row(2, "10-K", "2026-09-01", "annual.htm"),
        row(3, "8-K", "2026-10-03", "current.htm"), row(4, "4", "2026-10-05", "ownership.xml"),
    ]
    records[2]["reportDate"] = ""
    http = FakeHTTP({TICKERS_URL: json_response(TICKER_MAP), SUBMISSIONS_URL: json_response(submissions(records))})
    tool = SECResearchTool(client=http)
    assert isinstance(tool, ResearchTool)
    assert tool.spec == ResearchToolSpec(
        name="sec",
        description=("Search and retrieve official SEC EDGAR filings for a public company. "
                     "Use this tool when primary SEC filings such as 10-K, 10-Q, or 8-K are needed."),
        capabilities=("search", "read"), source_types=("10-K", "10-Q", "8-K"),
    )
    found = tool.search("exm")
    assert tool.search("EXM") == found
    assert [item["source_type"] for item in found] == ["8-K", "10-Q", "10-K"]
    assert [item["filing_date"] for item in found] == ["2026-10-03", "2026-10-01", "2026-09-01"]
    assert found[0]["report_date"] == ""
    assert found[1]["source_ref"] == SOURCE_REF
    assert found[1]["locator"] == ARCHIVE_URL
    assert found[1]["title"] == "Example Company 10-Q 2026-10-01"
    for item in found:
        assert {"source_ref", "title", "source_type", "locator"} <= item.keys()
        assert item["cik"] == "0001234567"
        assert item["company_name"] == "Example Company" and item["ticker"] == "EXM"
        assert item["form"] == item["source_type"]
        assert item["published_date"] == item["filing_date"]
        assert item["publisher"] == "U.S. Securities and Exchange Commission"
        assert item["primary_or_secondary"] == "Primary"
        assert item["independence_group"] == "sec:issuer:0001234567"
        assert "content" not in item and "spec" not in item
    assert [request.full_url for request, _, _ in http.calls] == [TICKERS_URL, SUBMISSIONS_URL] * 2
    for request, timeout, _ in http.calls:
        assert request.get_header("User-agent") == USER_AGENT
        assert request.get_header("Accept-encoding") == "gzip, deflate"
        assert timeout == 30.0
    times = [at for _, _, at in http.calls]
    assert all(right - left >= 0.1 for left, right in zip(times, times[1:]))
    registry = ResearchToolRegistry()
    mock = MockResearchTool()
    registry.register(mock)
    registry.register(tool)
    assert registry.get("sec") is tool
    assert registry.list_specs() == [mock.spec, tool.spec]


def test_search_limits_after_filtering_and_sorting():
    records = [row(n, filing_date=f"2026-10-{n:02d}") for n in range(1, 26)]
    records = [row(100 + n, "4", f"2026-11-{n:02d}", "ownership.xml") for n in range(1, 6)] + records
    http = FakeHTTP({TICKERS_URL: json_response(TICKER_MAP), SUBMISSIONS_URL: json_response(submissions(records))})
    result = SECResearchTool(client=http).search("EXM")
    assert len(result) == 20
    assert [item["filing_date"] for item in result] == [f"2026-10-{n:02d}" for n in range(25, 5, -1)]
    assert all(item["form"] == "10-Q" for item in result)


def test_read_resolves_encoded_reference_without_prior_search_and_preserves_html():
    document = "reports/quarter report.htm"
    source_ref = "sec:0001234567:0001234567-26-000001:reports%2Fquarter%20report.htm"
    url = "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/reports/quarter%20report.htm"
    raw = (
        "<ix:header><ix:hidden><ix:nonnumeric>HIDDEN META</ix:nonnumeric></ix:hidden></ix:header>"
        "<h1>Financial results</h1><p>Reported revenue was USD "
        "<ix:nonfraction>12</ix:nonfraction> million.</p>"
    )
    http = FakeHTTP({SUBMISSIONS_URL: json_response(submissions([row(document=document)])), url: FakeResponse(raw, "text/html; charset=utf-8")})
    material = SECResearchTool(client=http).read(source_ref)
    assert material["source_ref"] == source_ref
    assert material["locator"] == url
    assert material["content"] == "Financial results\n\nReported revenue was USD 12 million."
    assert material["raw_content"] == raw
    assert material["content_type"] == "text/html; charset=utf-8"
    assert material["title"] == "Example Company 10-Q 2026-10-01"
    assert material["ticker"] == "EXM" and material["report_date"] == "2026-06-30"
    assert material["primary_document"] == document
    assert material["publisher"] == "U.S. Securities and Exchange Commission"
    assert material["primary_or_secondary"] == "Primary"
    assert [request.full_url for request, _, _ in http.calls] == [SUBMISSIONS_URL, url]
    assert all(request.get_header("User-agent") == USER_AGENT and request.get_header("Accept-encoding") == "gzip, deflate"
               for request, _, _ in http.calls)


def test_html_conversion_preserves_blocks_and_table_text_without_scripts_or_styles():
    html = (
        "<style>.hidden {display:none}</style><script>doNotExtract()</script>"
        "<h2>Revenue &amp; growth</h2><p>First <b>reported</b> paragraph.</p>"
        "<table><tr><th>Metric</th><th>Amount</th></tr>"
        "<tr><td><p>Revenue</p><p>(USD millions)</p></td><td>12</td></tr></table>"
        "<p>Second paragraph.</p>"
    )
    expected = "Revenue & growth\n\nFirst reported paragraph.\n\nMetric Amount\nRevenue (USD millions) 12\n\nSecond paragraph."
    assert html_to_readable_text(html) == expected


@pytest.mark.parametrize("prefix", ["ix", "Facts"], ids=["standard-prefix", "declared-alias"])
def test_inline_xbrl_hidden_subtrees_removed_while_visible_text_and_blocks_survive(prefix):
    html = f"""
    <html xmlns:{prefix}="http://www.xbrl.org/2013/inlineXBRL">
      <header>Visible document header</header>
      <SCRIPT>HIDDEN SCRIPT</SCRIPT><STYLE>HIDDEN STYLE</STYLE>
      <{prefix}:HeAdEr>
        <{prefix}:references><link:schemaRef href="taxonomy.xsd"/>HIDDEN REFERENCES</{prefix}:references>
        <{prefix}:resources>
          <xbrli:context><xbrldi:explicitMember>us-gaap:HIDDEN MEMBER</xbrldi:explicitMember>
            <xbrli:period><xbrli:instant>2040-12-31</xbrli:instant></xbrli:period></xbrli:context>
          <xbrli:unit><xbrli:measure>HIDDEN UNIT</xbrli:measure></xbrli:unit>
        </{prefix}:resources>
        <{prefix}:hidden><{prefix}:nonnumeric>HIDDEN HEADER FACT</{prefix}:nonnumeric></{prefix}:hidden>
      </{prefix}:HeAdEr>
      <{prefix}:HiDdEn><{prefix}:nonnumeric>HIDDEN STANDALONE FACT</{prefix}:nonnumeric></{prefix}:HiDdEn>
      <div STYLE="color:red; DISPLAY : NoNe !important">
        <div>HIDDEN CONTAINER CHILD<br><img src="ignored.png"></div>
        HIDDEN AFTER CHILD
      </div>
      <h1>Company report</h1>
      <p style="font-weight:bold; display:inline"><{prefix}:nonNumeric>Example Company</{prefix}:nonNumeric> reported results.</p>
      <p>
        Revenue was
        <{prefix}:nonFraction name="us-gaap:Revenue">100</{prefix}:nonFraction>
        million.
      </p>
      <div style="color:blue">Visible div text.</div>
      <ul><li>First item</li><li>Second item</li></ul>
      <table><tr><th>Metric</th><th>Amount</th></tr>
        <tr><td>Revenue</td><td><{prefix}:nonfraction>100</{prefix}:nonfraction></td></tr>
        <tr><td>Costs</td><td>40</td></tr></table>
    </html>
    """
    normalized = html_to_readable_text(html)
    assert "HIDDEN" not in normalized and "2040-12-31" not in normalized
    assert normalized == (
        "Visible document header\n\nCompany report\n\nExample Company reported results.\n\n"
        "Revenue was 100 million.\n\nVisible div text.\n\nFirst item\n\nSecond item\n\n"
        "Metric Amount\nRevenue 100\nCosts 40"
    )
    assert html_to_readable_text(html) == normalized


def test_missing_user_agent_fails_before_http(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT")
    http = FakeHTTP({})
    with pytest.raises(SECConfigurationError, match="SEC_USER_AGENT"):
        SECResearchTool(client=http)
    assert http.calls == []


def test_unknown_ticker_is_an_explicit_error():
    http = FakeHTTP({TICKERS_URL: json_response(TICKER_MAP)})
    with pytest.raises(SECTickerNotFoundError, match="UNKNOWN"):
        SECResearchTool(client=http).search("unknown")
    assert len(http.calls) == 1


def test_natural_language_query_rejected_before_http():
    http = FakeHTTP({})
    with pytest.raises(SECResearchToolError, match="ticker"):
        SECResearchTool(client=http).search("latest Example 10-Q")
    assert not http.calls


@pytest.mark.parametrize("failure", [HTTPError(TICKERS_URL, 403, "Forbidden", {}, None), URLError("timeout")], ids=["http-error", "network-error"])
def test_http_errors_wrapped_without_retry(failure):
    http = FakeHTTP({TICKERS_URL: failure})
    with pytest.raises(SECResponseError) as error:
        SECResearchTool(client=http).search("EXM")
    assert error.value.__cause__ is failure
    assert len(http.calls) == 1


def test_invalid_json_raises_explicit_error():
    http = FakeHTTP({TICKERS_URL: FakeResponse("{not-json")})
    with pytest.raises(SECResponseError, match="Invalid SEC JSON"):
        SECResearchTool(client=http).search("EXM")


@pytest.mark.parametrize("failure", ["ticker-map", "recent-array-length", "missing-primary-document"])
def test_malformed_metadata_is_not_silently_ignored(failure):
    tickers, payload = deepcopy(TICKER_MAP), submissions([row()])
    if failure == "ticker-map":
        del tickers["0"]["cik_str"]
    elif failure == "recent-array-length":
        payload["filings"]["recent"]["accessionNumber"] = []
    else:
        payload["filings"]["recent"]["primaryDocument"] = [""]
    http = FakeHTTP({TICKERS_URL: json_response(tickers), SUBMISSIONS_URL: json_response(payload)})
    with pytest.raises(SECResponseError):
        SECResearchTool(client=http).search("EXM")


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
def test_compressed_plain_text_preserved(encoding):
    raw = "Heading\n\nA source-stated fact.\n"
    compressed = (gzip.compress if encoding == "gzip" else zlib.compress)(raw.encode("utf-8"))
    url = ARCHIVE_URL.replace("report.htm", "report.txt")
    http = FakeHTTP({
        SUBMISSIONS_URL: json_response(submissions([row(document="report.txt")])),
        url: FakeResponse(compressed, "text/plain; charset=utf-8", encoding),
    })
    material = SECResearchTool(client=http).read(SOURCE_REF.replace("report.htm", "report.txt"))
    assert material["content"] == material["raw_content"] == raw


@pytest.mark.parametrize("raw", [" \n\t", "<script>ignored()</script><style>p {color:red}</style>"])
def test_empty_document_content_rejected(raw):
    http = FakeHTTP({SUBMISSIONS_URL: json_response(submissions([row()])), ARCHIVE_URL: FakeResponse(raw, "text/html")})
    with pytest.raises(SECResponseError, match="Empty SEC primary document"):
        SECResearchTool(client=http).read(SOURCE_REF)


def test_unknown_recent_filing_does_not_fetch_a_document():
    http = FakeHTTP({SUBMISSIONS_URL: json_response(submissions([row(2)]))})
    with pytest.raises(SECResponseError, match="supported recent filing"):
        SECResearchTool(client=http).read(SOURCE_REF)
    assert len(http.calls) == 1


def test_invalid_source_ref_rejected_before_http():
    http = FakeHTTP({})
    with pytest.raises(SECResearchToolError, match="source_ref"):
        SECResearchTool(client=http).read(ARCHIVE_URL)
    assert not http.calls
