#!/usr/bin/env python3
"""
Multimodal Swiss Document Dataset Collector for VLM Fine-Tuning.

Scrapes public Swiss data from multiple sources and structures it into
a HuggingFace Dataset with train/val/test splits.

Sources:
  - SBB (Swiss Federal Railways) timetables and station images
  - ZVV (Zürich Transport Network) PDFs
  - admin.ch Swiss government forms (PDF)
  - swissinfo.ch / nzz.ch news articles with images
  - Migros / Coop product catalog images
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import httpx
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from PIL import Image as PILImage
from PyPDF2 import PdfReader
import pdfplumber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("swiss-data")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "swiss_dataset"
IMAGE_DIR = DATA_DIR / "images"
PDF_DIR = DATA_DIR / "pdfs"
RAW_DIR = DATA_DIR / "raw"

for d in (IMAGE_DIR, PDF_DIR, RAW_DIR):
    d.mkdir(parents=True, exist_ok=True)

HTTPX_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-CH,de;q=0.9,en;q=0.8",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    question: str
    answer: str


@dataclass
class DatasetExample:
    image_path: str
    text: str
    language: str
    source: str
    category: str
    qa_pairs: list[QAPair] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


def save_image(content: bytes, name_hint: str) -> str:
    """Save raw image bytes and return the relative path."""
    ext = _detect_image_ext(content)
    filename = f"{hashlib.md5(content).hexdigest()[:12]}_{slug(name_hint)}{ext}"
    path = IMAGE_DIR / filename
    path.write_bytes(content)
    return str(path.relative_to(BASE_DIR))


def _detect_image_ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"GIF8":
        return ".gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def download_image(client: httpx.Client, url: str, name_hint: str) -> Optional[str]:
    try:
        resp = client.get(url, follow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "image" not in ct and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            logger.warning("Not an image: %s (content-type=%s)", url, ct)
            return None
        return save_image(resp.content, name_hint)
    except Exception as exc:
        logger.warning("Failed to download image %s: %s", url, exc)
        return None


def download_pdf(client: httpx.Client, url: str, name_hint: str) -> Optional[Path]:
    try:
        resp = client.get(url, follow_redirects=True)
        resp.raise_for_status()
        filename = f"{hashlib.md5(resp.content).hexdigest()[:12]}_{slug(name_hint)}.pdf"
        path = PDF_DIR / filename
        path.write_bytes(resp.content)
        return path
    except Exception as exc:
        logger.warning("Failed to download PDF %s: %s", url, exc)
        return None


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pdfplumber with PyPDF2 fallback."""
    text_parts: list[str] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
    except Exception:
        try:
            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        except Exception as exc:
            logger.warning("Could not extract text from %s: %s", pdf_path, exc)
    return "\n".join(text_parts).strip()


def pdf_first_page_image(pdf_path: Path, name_hint: str) -> Optional[str]:
    """Render the first page of a PDF to an image and save it."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if not pdf.pages:
                return None
            page = pdf.pages[0]
            img = page.to_image(resolution=150)
            buf = BytesIO()
            img.save(buf, format="PNG")
            return save_image(buf.getvalue(), name_hint)
    except Exception as exc:
        logger.warning("Could not render PDF page %s: %s", pdf_path, exc)
        return None


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------

class SBBScraper:
    """Scrape SBB (Swiss Federal Railways) station images and timetable data."""

    # Public SBB OpenData endpoints (no auth required for these)
    STATION_SEARCH_URL = "https://transport.opendata.ch/v1/locations"
    CONNECTIONS_URL = "https://transport.opendata.ch/v1/connections"
    STATIONS = [
        "Zürich HB", "Bern", "Basel SBB", "Genève", "Lausanne",
        "Luzern", "Lugano", "St. Gallen", "Interlaken Ost", "Chur",
        "Bellinzona", "Thun", "Biel/Bienne", "Schaffhausen", "Winterthur",
    ]

    def collect(self, client: httpx.Client, limit: int = 120) -> list[DatasetExample]:
        examples: list[DatasetExample] = []
        logger.info("SBB: collecting station & timetable data …")

        # 1. Station info pairs
        for station_name in self.STATIONS:
            if len(examples) >= limit:
                break
            try:
                resp = client.get(
                    self.STATION_SEARCH_URL,
                    params={"query": station_name, "limit": 1},
                )
                resp.raise_for_status()
                data = resp.json()
                stations = data.get("stations", [])
                if not stations:
                    continue
                st = stations[0]
                text_parts = [f"Station: {st.get('name', station_name)}"]
                if st.get("id"):
                    text_parts.append(f"Station ID: {st['id']}")
                if st.get("coordinate"):
                    c = st["coordinate"]
                    text_parts.append(
                        f"Coordinates: {c.get('x')}, {c.get('y')}"
                    )
                text = "\n".join(text_parts)

                # Download station image from Wikipedia/Wikimedia
                img_path = self._fetch_station_image(client, station_name)

                qa = [
                    QAPair(
                        question=f"What is the station ID of {station_name}?",
                        answer=st.get("id", "Unknown"),
                    ),
                    QAPair(
                        question=f"Where is {station_name} located?",
                        answer=(
                            f"Coordinates: {st.get('coordinate', {}).get('x', 'N/A')}, "
                            f"{st.get('coordinate', {}).get('y', 'N/A')}"
                        ),
                    ),
                ]
                examples.append(DatasetExample(
                    image_path=img_path or "",
                    text=text,
                    language="de",
                    source="sbb",
                    category="transport",
                    qa_pairs=qa,
                    metadata={"station": station_name, "station_id": st.get("id")},
                ))
            except Exception as exc:
                logger.warning("SBB station %s failed: %s", station_name, exc)

        # 2. Timetable / connections between random station pairs
        import random
        pairs = [(a, b) for a in self.STATIONS for b in self.STATIONS if a != b]
        random.shuffle(pairs)
        for fro, to in pairs:
            if len(examples) >= limit:
                break
            try:
                resp = client.get(
                    self.CONNECTIONS_URL,
                    params={"from": fro, "to": to, "limit": 3},
                )
                resp.raise_for_status()
                data = resp.json()
                conns = data.get("connections", [])
                if not conns:
                    continue
                lines = [f"Connections from {fro} to {to}:"]
                for conn in conns[:3]:
                    dep = conn.get("from", {}).get("departure", "?")
                    arr = conn.get("to", {}).get("arrival", "?")
                    duration = conn.get("duration", "?")
                    platform = conn.get("from", {}).get("platform", "?")
                    lines.append(
                        f"  Depart {dep} → Arrive {arr}  (duration {duration}, platform {platform})"
                    )
                    sections = conn.get("sections", [])
                    for sec in sections:
                        journey = sec.get("journey")
                        if journey:
                            lines.append(
                                f"    Train {journey.get('number', '?')} "
                                f"({journey.get('category', '?')}) "
                                f"direction {journey.get('to', '?')}"
                            )
                text = "\n".join(lines)

                qa = [
                    QAPair(
                        question=f"When does the first train from {fro} to {to} depart?",
                        answer=conns[0].get("from", {}).get("departure", "Unknown"),
                    ),
                    QAPair(
                        question=f"How long is the journey from {fro} to {to}?",
                        answer=conns[0].get("duration", "Unknown"),
                    ),
                ]
                examples.append(DatasetExample(
                    image_path="",
                    text=text,
                    language="de",
                    source="sbb",
                    category="timetable",
                    qa_pairs=qa,
                    metadata={"from": fro, "to": to, "num_connections": len(conns)},
                ))
            except Exception as exc:
                logger.warning("SBB connection %s→%s failed: %s", fro, to, exc)

        logger.info("SBB: collected %d examples", len(examples))
        return examples

    def _fetch_station_image(self, client: httpx.Client, station: str) -> Optional[str]:
        """Try to fetch a station image from Wikimedia Commons."""
        try:
            search_url = "https://commons.wikimedia.org/w/api.php"
            resp = client.get(search_url, params={
                "action": "query",
                "list": "search",
                "srsearch": f"{station} Bahnhof railway station",
                "srnamespace": 6,
                "srlimit": 1,
                "format": "json",
            })
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            if not results:
                return None
            title = results[0]["title"]
            # Get image info
            resp2 = client.get(search_url, params={
                "action": "query",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url",
                "iiurlwidth": 800,
                "format": "json",
            })
            resp2.raise_for_status()
            pages = resp2.json().get("query", {}).get("pages", {})
            for page in pages.values():
                ii = page.get("imageinfo", [{}])[0]
                url = ii.get("thumburl") or ii.get("url")
                if url:
                    return download_image(client, url, station)
        except Exception as exc:
            logger.debug("Wikimedia image fetch failed for %s: %s", station, exc)
        return None


class ZVVScraper:
    """Scrape ZVV (Zürich Transport Network) PDF network plans and maps."""

    ZVV_PDFS = [
        {
            "url": "https://www.zvv.ch/zvv-assets/liniennetz/zonenplan_gesamt.pdf",
            "name": "ZVV zone map",
            "category": "zone_map",
        },
        {
            "url": "https://www.zvv.ch/zvv-assets/liniennetz/liniennetzplan.pdf",
            "name": "ZVV network plan",
            "category": "network_plan",
        },
    ]

    # Additional ZVV-related endpoints
    ZVV_API = "https://api.opentransportdata.swiss/la/connections"

    def collect(self, client: httpx.Client, limit: int = 80) -> list[DatasetExample]:
        examples: list[DatasetExample] = []
        logger.info("ZVV: collecting transport PDFs and maps …")

        for pdf_info in self.ZVV_PDFS:
            if len(examples) >= limit:
                break
            pdf_path = download_pdf(client, pdf_info["url"], pdf_info["name"])
            if not pdf_path:
                continue
            text = extract_pdf_text(pdf_path)
            img_path = pdf_first_page_image(pdf_path, pdf_info["name"])
            qa = self._generate_zvv_qa(text, pdf_info["category"])
            examples.append(DatasetExample(
                image_path=img_path or "",
                text=text[:3000],
                language="de",
                source="zvv",
                category=pdf_info["category"],
                qa_pairs=qa,
                metadata={"pdf_path": str(pdf_path.relative_to(BASE_DIR))},
            ))

        # Scrape ZVV stop/line data from transport.opendata.ch
        zurich_stops = [
            "Zürich, Stadelhofen", "Zürich, Oerlikon", "Zürich, Hardbrücke",
            "Zürich, Wiedikon", "Zürich, Tiefenbrunnen", "Zürich, Altstetten",
            "Zürich, Enge", "Zürich, Wollishofen", "Zürich, Limmatplatz",
            "Zürich, Bellevue", "Zürich, Bürkliplatz", "Zürich, Paradeplatz",
        ]
        for stop in zurich_stops:
            if len(examples) >= limit:
                break
            try:
                resp = client.get(
                    "https://transport.opendata.ch/v1/locations",
                    params={"query": stop, "type": "station", "limit": 1},
                )
                resp.raise_for_status()
                data = resp.json()
                stations = data.get("stations", [])
                if not stations:
                    continue
                st = stations[0]
                lines_list = st.get("lines", [])
                text_parts = [f"Stop: {st.get('name', stop)}"]
                if lines_list:
                    text_parts.append("Lines serving this stop:")
                    for ln in lines_list[:15]:
                        text_parts.append(
                            f"  {ln.get('name', '?')} ({ln.get('category', '?')}) "
                            f"→ {ln.get('to', '?')}"
                        )
                text = "\n".join(text_parts)
                qa = [
                    QAPair(
                        question=f"What transport lines serve {stop}?",
                        answer=", ".join(
                            ln.get("name", "?") for ln in lines_list[:5]
                        ) or "Unknown",
                    ),
                    QAPair(
                        question=f"Where is {stop} located?",
                        answer=(
                            f"Lat: {st.get('coordinate', {}).get('x', 'N/A')}, "
                            f"Lon: {st.get('coordinate', {}).get('y', 'N/A')}"
                        ),
                    ),
                ]
                examples.append(DatasetExample(
                    image_path="",
                    text=text,
                    language="de",
                    source="zvv",
                    category="stop_info",
                    qa_pairs=qa,
                    metadata={"stop": stop, "station_id": st.get("id")},
                ))
            except Exception as exc:
                logger.warning("ZVV stop %s failed: %s", stop, exc)

        logger.info("ZVV: collected %d examples", len(examples))
        return examples

    def _generate_zvv_qa(self, text: str, category: str) -> list[QAPair]:
        if category == "zone_map":
            return [
                QAPair(
                    question="What does this document show?",
                    answer="This is a ZVV zone map showing the fare zones for public transport in the Zürich area.",
                ),
                QAPair(
                    question="How many zones are in the ZVV network?",
                    answer=self._extract_zone_count(text),
                ),
            ]
        return [
            QAPair(
                question="What does this document show?",
                answer="This is a ZVV network plan showing tram, bus, and train lines in the Zürich area.",
            ),
        ]

    @staticmethod
    def _extract_zone_count(text: str) -> str:
        nums = re.findall(r"\bzone\s*(\d+)", text, re.IGNORECASE)
        if nums:
            return f"At least {max(int(n) for n in nums)} zones"
        return "See document for zone details"


class AdminCHScraper:
    """Scrape Swiss government forms and documents from admin.ch."""

    # Public PDF forms available on admin.ch
    FORM_URLS = [
        {
            "url": "https://www.bj.admin.ch/dam/data/bj/gesellschaft/gesetzgebung/ahv/aend-ahv/merkblatt-d.pdf",
            "name": "AHV Merkblatt",
            "lang": "de",
            "category": "social_insurance",
        },
        {
            "url": "https://www.sem.admin.ch/dam/data/sem/einreise/merkblaetter/merkblatt-einreise-d.pdf",
            "name": "Einreise Merkblatt",
            "lang": "de",
            "category": "immigration",
        },
        {
            "url": "https://www.edi.admin.ch/dam/edi/de/dokumente/gleichstellung/merkblatt-behindertenrecht.pdf",
            "name": "Behindertenrecht Merkblatt",
            "lang": "de",
            "category": "disability_rights",
        },
    ]

    def collect(self, client: httpx.Client, limit: int = 80) -> list[DatasetExample]:
        examples: list[DatasetExample] = []
        logger.info("admin.ch: collecting government forms …")

        for form_info in self.FORM_URLS:
            if len(examples) >= limit:
                break
            pdf_path = download_pdf(client, form_info["url"], form_info["name"])
            if not pdf_path:
                continue
            text = extract_pdf_text(pdf_path)
            if not text:
                logger.warning("Empty text from %s", pdf_path)
                continue
            img_path = pdf_first_page_image(pdf_path, form_info["name"])
            qa = self._generate_form_qa(text, form_info["category"])
            examples.append(DatasetExample(
                image_path=img_path or "",
                text=text[:4000],
                language=form_info["lang"],
                source="admin.ch",
                category=form_info["category"],
                qa_pairs=qa,
                metadata={
                    "form_name": form_info["name"],
                    "pdf_path": str(pdf_path.relative_to(BASE_DIR)),
                },
            ))

        # Scrape admin.ch for additional document links
        self._scrape_admin_index(client, examples, limit)

        logger.info("admin.ch: collected %d examples", len(examples))
        return examples

    def _scrape_admin_index(
        self, client: httpx.Client, examples: list[DatasetExample], limit: int
    ) -> None:
        """Try to discover additional PDF forms from admin.ch."""
        try:
            resp = client.get(
                "https://www.admin.ch/gov/en/start/documentation/forms.html",
                follow_redirects=True,
            )
            resp.raise_for_status()
            from html.parser import HTMLParser

            class LinkExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.links: list[str] = []
                def handle_starttag(self, tag, attrs):
                    if tag == "a":
                        for name, val in attrs:
                            if name == "href" and val and val.endswith(".pdf"):
                                self.links.append(val)

            parser = LinkExtractor()
            parser.feed(resp.text)
            seen = {fi["url"] for fi in self.FORM_URLS}
            for link in parser.links[:10]:
                if len(examples) >= limit:
                    break
                if link in seen:
                    continue
                if not link.startswith("http"):
                    link = f"https://www.admin.ch{link}"
                seen.add(link)
                name = link.rsplit("/", 1)[-1].replace(".pdf", "")
                pdf_path = download_pdf(client, link, name)
                if not pdf_path:
                    continue
                text = extract_pdf_text(pdf_path)
                if not text:
                    continue
                img_path = pdf_first_page_image(pdf_path, name)
                qa = self._generate_form_qa(text, "government_form")
                examples.append(DatasetExample(
                    image_path=img_path or "",
                    text=text[:4000],
                    language="de",
                    source="admin.ch",
                    category="government_form",
                    qa_pairs=qa,
                    metadata={
                        "form_name": name,
                        "pdf_path": str(pdf_path.relative_to(BASE_DIR)),
                    },
                ))
        except Exception as exc:
            logger.debug("admin.ch index scrape failed: %s", exc)

    def _generate_form_qa(self, text: str, category: str) -> list[QAPair]:
        # Extract key entities from text
        date_matches = re.findall(
            r"\d{1,2}\.\d{1,2}\.\d{4}", text
        )
        qa = [
            QAPair(
                question="What type of Swiss government document is this?",
                answer=f"This is a {category.replace('_', ' ')} document from the Swiss federal administration.",
            ),
        ]
        if date_matches:
            qa.append(QAPair(
                question="What dates are mentioned in this document?",
                answer=", ".join(date_matches[:5]),
            ))
        # Try to extract a heading
        heading = re.search(r"^(.{10,80})$", text, re.MULTILINE)
        if heading:
            qa.append(QAPair(
                question="What is the title of this document?",
                answer=heading.group(1).strip(),
            ))
        return qa


class NewsScraper:
    """Scrape Swiss news articles with images from swissinfo.ch and nzz.ch."""

    SWISSINFO_API = "https://www.swissinfo.ch/eng/api/v3/search"
    NZZ_SITEMAP = "https://www.nzz.ch/sitemap.xml"

    def collect(self, client: httpx.Client, limit: int = 120) -> list[DatasetExample]:
        examples: list[DatasetExample] = []
        logger.info("News: collecting articles from swissinfo & nzz …")

        # SwissInfo RSS / API
        self._collect_swissinfo(client, examples, limit // 2)

        # NZZ
        self._collect_nzz(client, examples, limit)

        logger.info("News: collected %d examples", len(examples))
        return examples

    def _collect_swissinfo(
        self, client: httpx.Client, examples: list[DatasetExample], limit: int
    ) -> None:
        """Scrape swissinfo.ch articles via their public search API."""
        topics = [
            "swiss politics", "economy", "society", "culture",
            "science", "switzerland", "alps", "banking",
        ]
        for topic in topics:
            if len(examples) >= limit:
                break
            try:
                resp = client.get(
                    self.SWISSINFO_API,
                    params={"query": topic, "limit": 5, "language": "en"},
                    follow_redirects=True,
                )
                resp.raise_for_status()
                articles = resp.json().get("results", resp.json().get("data", []))
                if not articles:
                    # Fallback: scrape HTML search page
                    articles = self._scrape_swissinfo_html(client, topic)
                for article in articles[:5]:
                    if len(examples) >= limit:
                        break
                    title = article.get("title", article.get("name", ""))
                    body = article.get("body", article.get("text", article.get("lead", "")))
                    image_url = article.get("imageUrl", article.get("image", {}).get("url", ""))
                    article_url = article.get("url", "")

                    text_parts = [f"Title: {title}"]
                    if body:
                        text_parts.append(body[:2000])
                    if article_url:
                        text_parts.append(f"Source: {article_url}")
                    text = "\n\n".join(text_parts)

                    img_path = ""
                    if image_url:
                        img_path = download_image(client, image_url, title) or ""

                    qa = [
                        QAPair(
                            question="What is this article about?",
                            answer=title or "Swiss news article",
                        ),
                        QAPair(
                            question="What is the main topic?",
                            answer=topic,
                        ),
                    ]
                    examples.append(DatasetExample(
                        image_path=img_path,
                        text=text,
                        language="en",
                        source="swissinfo.ch",
                        category="news",
                        qa_pairs=qa,
                        metadata={"topic": topic, "url": article_url},
                    ))
            except Exception as exc:
                logger.warning("swissinfo topic '%s' failed: %s", topic, exc)

    def _scrape_swissinfo_html(self, client: httpx.Client, query: str) -> list[dict]:
        """Fallback: parse swissinfo.ch search HTML for article data."""
        articles = []
        try:
            resp = client.get(
                f"https://www.swissinfo.ch/eng/search",
                params={"query": query},
                follow_redirects=True,
            )
            resp.raise_for_status()
            from html.parser import HTMLParser

            class ArticleExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._in_article = False
                    self._current: dict = {}
                    self.articles: list[dict] = []
                    self._tag_stack: list[str] = []

                def handle_starttag(self, tag, attrs):
                    attrs_d = dict(attrs)
                    cls = attrs_d.get("class", "")
                    if tag == "article" or "teaser" in cls:
                        self._in_article = True
                        self._current = {}
                    if self._in_article and tag == "a":
                        href = attrs_d.get("href", "")
                        if href and "/eng/" in href:
                            self._current.setdefault("url", href)
                    if self._in_article and tag == "img":
                        src = attrs_d.get("src") or attrs_d.get("data-src", "")
                        if src:
                            self._current["imageUrl"] = src
                    self._tag_stack.append(tag)

                def handle_data(self, data):
                    if self._in_article and self._tag_stack:
                        text = data.strip()
                        if text and len(text) > 10:
                            self._current.setdefault("title", text)

                def handle_endtag(self, tag):
                    if self._tag_stack:
                        self._tag_stack.pop()
                    if self._in_article and tag == "article":
                        if self._current.get("title"):
                            self.articles.append(self._current)
                        self._current = {}
                        self._in_article = False

            parser = ArticleExtractor()
            parser.feed(resp.text)
            articles = parser.articles[:5]
        except Exception as exc:
            logger.debug("swissinfo HTML parse failed: %s", exc)
        return articles

    def _collect_nzz(
        self, client: httpx.Client, examples: list[DatasetExample], limit: int
    ) -> None:
        """Scrape NZZ articles from their sitemap."""
        try:
            resp = client.get(self.NZZ_SITEMAP, follow_redirects=True)
            resp.raise_for_status()
            # Extract article URLs from sitemap XML
            urls = re.findall(
                r"<loc>(https://www\.nzz\.ch/[^<]+)</loc>", resp.text
            )
            # Filter for article URLs (typically have numeric IDs or long paths)
            article_urls = [
                u for u in urls
                if re.search(r"ld\.\d+", u) or len(u.split("/")) > 5
            ][:30]

            for url in article_urls:
                if len(examples) >= limit:
                    break
                try:
                    resp2 = client.get(url, follow_redirects=True, timeout=15)
                    resp2.raise_for_status()
                    title_match = re.search(
                        r"<title[^>]*>([^<]+)</title>", resp2.text
                    )
                    title = title_match.group(1).strip() if title_match else ""
                    # Extract og:image
                    img_match = re.search(
                        r'og:image["\s]+content="([^"]+)"', resp2.text
                    )
                    img_url = img_match.group(1) if img_match else ""
                    # Extract article text from JSON-LD
                    jsonld_match = re.search(
                        r'<script type="application/ld\+json">(.*?)</script>',
                        resp2.text, re.DOTALL,
                    )
                    body = ""
                    if jsonld_match:
                        try:
                            ld = json.loads(jsonld_match.group(1))
                            body = ld.get("articleBody", "")
                        except json.JSONDecodeError:
                            pass
                    if not body:
                        # Fallback: strip HTML tags roughly
                        body_match = re.search(
                            r"<article[^>]*>(.*?)</article>", resp2.text, re.DOTALL
                        )
                        if body_match:
                            body = re.sub(r"<[^>]+>", " ", body_match.group(1))
                            body = re.sub(r"\s+", " ", body).strip()

                    text_parts = [f"Title: {title}"]
                    if body:
                        text_parts.append(body[:2000])
                    text_parts.append(f"Source: {url}")
                    text = "\n\n".join(text_parts)

                    img_path = ""
                    if img_url:
                        img_path = download_image(client, img_url, title) or ""

                    qa = [
                        QAPair(
                            question="What is this article about?",
                            answer=title or "Swiss news article from NZZ",
                        ),
                    ]
                    examples.append(DatasetExample(
                        image_path=img_path,
                        text=text,
                        language="de",
                        source="nzz.ch",
                        category="news",
                        qa_pairs=qa,
                        metadata={"url": url},
                    ))
                except Exception as exc:
                    logger.debug("NZZ article %s failed: %s", url, exc)
        except Exception as exc:
            logger.warning("NZZ sitemap fetch failed: %s", exc)


class ProductCatalogScraper:
    """Scrape Swiss product catalog images from Migros and Coop."""

    MIGROS_CATEGORIES = [
        "https://www.migros.ch/de/category/1000",
        "https://www.migros.ch/de/category/2000",
        "https://www.migros.ch/de/category/3000",
        "https://www.migros.ch/de/category/4000",
        "https://www.migros.ch/de/category/5000",
        "https://www.migros.ch/de/category/6000",
    ]
    COOP_SEARCH = "https://www.coop.ch/en/search/"

    def collect(self, client: httpx.Client, limit: int = 100) -> list[DatasetExample]:
        examples: list[DatasetExample] = []
        logger.info("Products: collecting Migros & Coop catalog data …")

        self._collect_migros(client, examples, limit // 2)
        self._collect_coop(client, examples, limit)

        logger.info("Products: collected %d examples", len(examples))
        return examples

    def _collect_migros(
        self, client: httpx.Client, examples: list[DatasetExample], limit: int
    ) -> None:
        for cat_url in self.MIGROS_CATEGORIES:
            if len(examples) >= limit:
                break
            try:
                resp = client.get(cat_url, follow_redirects=True)
                resp.raise_for_status()
                # Parse product cards from HTML
                products = self._parse_migros_products(resp.text, cat_url)
                for prod in products[:10]:
                    if len(examples) >= limit:
                        break
                    img_path = ""
                    if prod.get("image_url"):
                        img_path = download_image(
                            client, prod["image_url"], prod["name"]
                        ) or ""

                    text = f"Product: {prod['name']}"
                    if prod.get("price"):
                        text += f"\nPrice: {prod['price']}"
                    if prod.get("brand"):
                        text += f"\nBrand: {prod['brand']}"
                    text += f"\nRetailer: Migros"
                    text += f"\nCategory: {prod.get('category', 'N/A')}"

                    qa = [
                        QAPair(
                            question=f"What is the price of {prod['name']}?",
                            answer=prod.get("price", "Price not available"),
                        ),
                        QAPair(
                            question=f"Which retailer sells {prod['name']}?",
                            answer="Migros",
                        ),
                    ]
                    examples.append(DatasetExample(
                        image_path=img_path,
                        text=text,
                        language="de",
                        source="migros.ch",
                        category="product",
                        qa_pairs=qa,
                        metadata={
                            "product_name": prod["name"],
                            "brand": prod.get("brand", ""),
                            "url": prod.get("url", ""),
                        },
                    ))
            except Exception as exc:
                logger.warning("Migros category %s failed: %s", cat_url, exc)

    def _parse_migros_products(self, html: str, base_url: str) -> list[dict]:
        """Parse Migros product cards from HTML."""
        products = []
        # Look for JSON-LD product data
        for match in re.finditer(
            r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL
        ):
            try:
                ld = json.loads(match.group(1))
                if isinstance(ld, list):
                    for item in ld:
                        if item.get("@type") == "Product":
                            products.append({
                                "name": item.get("name", ""),
                                "price": (
                                    item.get("offers", {}).get("price", "")
                                    + " "
                                    + item.get("offers", {}).get("priceCurrency", "CHF")
                                ),
                                "brand": item.get("brand", {}).get("name", ""),
                                "image_url": (
                                    item.get("image", [""])[0]
                                    if isinstance(item.get("image"), list)
                                    else item.get("image", "")
                                ),
                                "url": item.get("url", ""),
                                "category": item.get("category", ""),
                            })
                elif ld.get("@type") == "Product":
                    products.append({
                        "name": ld.get("name", ""),
                        "price": (
                            ld.get("offers", {}).get("price", "")
                            + " "
                            + ld.get("offers", {}).get("priceCurrency", "CHF")
                        ),
                        "brand": ld.get("brand", {}).get("name", ""),
                        "image_url": (
                            ld.get("image", [""])[0]
                            if isinstance(ld.get("image"), list)
                            else ld.get("image", "")
                        ),
                        "url": ld.get("url", ""),
                        "category": ld.get("category", ""),
                    })
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: regex-based extraction
        if not products:
            for match in re.finditer(
                r'data-product-name="([^"]+)".*?'
                r'(?:data-price="([^"]*)")?.*?'
                r'(?:src|data-src)="([^"]*\.(?:jpg|png|webp)[^"]*)"',
                html, re.DOTALL,
            ):
                products.append({
                    "name": match.group(1),
                    "price": match.group(2) or "",
                    "image_url": match.group(3),
                    "brand": "",
                    "url": "",
                    "category": "",
                })
        return products[:20]

    def _collect_coop(
        self, client: httpx.Client, examples: list[DatasetExample], limit: int
    ) -> None:
        search_terms = [
            "milk", "bread", "chocolate", "cheese", "wine",
            "yogurt", "pasta", "coffee", "butter", "eggs",
        ]
        for term in search_terms:
            if len(examples) >= limit:
                break
            try:
                resp = client.get(
                    self.COOP_SEARCH,
                    params={"q": term, "searchTerm": term},
                    follow_redirects=True,
                )
                resp.raise_for_status()
                products = self._parse_coop_products(resp.text)
                for prod in products[:5]:
                    if len(examples) >= limit:
                        break
                    img_path = ""
                    if prod.get("image_url"):
                        img_path = download_image(
                            client, prod["image_url"], prod["name"]
                        ) or ""
                    text = f"Product: {prod['name']}"
                    if prod.get("price"):
                        text += f"\nPrice: {prod['price']}"
                    text += f"\nRetailer: Coop"
                    text += f"\nSearch term: {term}"

                    qa = [
                        QAPair(
                            question=f"What is the price of {prod['name']} at Coop?",
                            answer=prod.get("price", "Price not available"),
                        ),
                        QAPair(
                            question=f"Which Swiss retailer sells {prod['name']}?",
                            answer="Coop",
                        ),
                    ]
                    examples.append(DatasetExample(
                        image_path=img_path,
                        text=text,
                        language="en",
                        source="coop.ch",
                        category="product",
                        qa_pairs=qa,
                        metadata={
                            "product_name": prod["name"],
                            "search_term": term,
                        },
                    ))
            except Exception as exc:
                logger.warning("Coop search '%s' failed: %s", term, exc)

    def _parse_coop_products(self, html: str) -> list[dict]:
        products = []
        # JSON-LD
        for match in re.finditer(
            r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL
        ):
            try:
                ld = json.loads(match.group(1))
                items = ld if isinstance(ld, list) else [ld]
                for item in items:
                    if item.get("@type") == "Product":
                        products.append({
                            "name": item.get("name", ""),
                            "price": str(item.get("offers", {}).get("price", "")),
                            "image_url": (
                                item.get("image", [""])[0]
                                if isinstance(item.get("image"), list)
                                else item.get("image", "")
                            ),
                        })
            except (json.JSONDecodeError, TypeError):
                pass

        # Regex fallback
        if not products:
            for match in re.finditer(
                r'product-card.*?<(?:h[23]|span)[^>]*class="[^"]*name[^"]*"[^>]*>'
                r'([^<]+)</(?:h[23]|span)>.*?'
                r'(?:price[^>]*>([^<]*)<)?.*?'
                r'(?:src|data-src)="([^"]*\.(?:jpg|png|webp)[^"]*)"',
                html, re.DOTALL,
            ):
                products.append({
                    "name": match.group(1).strip(),
                    "price": match.group(2).strip() if match.group(2) else "",
                    "image_url": match.group(3),
                })
        return products[:15]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class SwissDatasetBuilder:
    """Assemble collected examples into a HuggingFace Dataset with splits."""

    def __init__(self, examples: list[DatasetExample]):
        self.examples = examples

    def build(self, output_dir: Optional[Path] = None) -> DatasetDict:
        if output_dir is None:
            output_dir = DATA_DIR / "dataset"

        # Shuffle and split: 80% train, 10% val, 10% test
        import random
        random.seed(42)
        random.shuffle(self.examples)
        n = len(self.examples)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        splits = {
            "train": self.examples[:n_train],
            "val": self.examples[n_train:n_train + n_val],
            "test": self.examples[n_train + n_val:],
        }

        dataset_dict = {}
        for split_name, split_examples in splits.items():
            records = self._to_records(split_examples)
            features = Features({
                "image": Image(),
                "text": Value("string"),
                "language": Value("string"),
                "source": Value("string"),
                "category": Value("string"),
                "qa_pairs": Sequence({
                    "question": Value("string"),
                    "answer": Value("string"),
                }),
                "metadata": Value("string"),
            })
            ds = Dataset.from_list(records, features=features)
            dataset_dict[split_name] = ds

        dd = DatasetDict(dataset_dict)

        output_dir.mkdir(parents=True, exist_ok=True)
        dd.save_to_disk(str(output_dir))
        logger.info("Dataset saved to %s", output_dir)

        # Also save metadata JSON
        meta_path = output_dir / "metadata.json"
        meta = {
            "total_examples": n,
            "splits": {k: len(v) for k, v in splits.items()},
            "sources": list(set(e.source for e in self.examples)),
            "categories": list(set(e.category for e in self.examples)),
            "languages": list(set(e.language for e in self.examples)),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("Metadata saved to %s", meta_path)

        return dd

    def _to_records(self, examples: list[DatasetExample]) -> list[dict]:
        records = []
        for ex in examples:
            img = None
            if ex.image_path:
                abs_path = BASE_DIR / ex.image_path
                if abs_path.exists():
                    img = str(abs_path)
            qa_list = [asdict(qa) for qa in ex.qa_pairs]
            qa_pairs_seq = {
                "question": [q["question"] for q in qa_list],
                "answer": [q["answer"] for q in qa_list],
            }
            records.append({
                "image": img,
                "text": ex.text,
                "language": ex.language,
                "source": ex.source,
                "category": ex.category,
                "qa_pairs": qa_pairs_seq,
                "metadata": json.dumps(ex.metadata),
            })
        return records


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect Swiss public data for VLM fine-tuning"
    )
    parser.add_argument(
        "--limit", type=int, default=800,
        help="Target total number of examples (default: 800)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for HuggingFace dataset",
    )
    parser.add_argument(
        "--scrapers", nargs="+",
        choices=["sbb", "zvv", "admin", "news", "products", "all"],
        default=["all"],
        help="Which scrapers to run",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay in seconds between HTTP requests",
    )
    args = parser.parse_args()

    scrapers_to_run = set(args.scrapers)
    if "all" in scrapers_to_run:
        scrapers_to_run = {"sbb", "zvv", "admin", "news", "products"}

    # Divide limit across scrapers
    per_scraper = max(args.limit // len(scrapers_to_run), 50)

    all_examples: list[DatasetExample] = []

    with httpx.Client(
        timeout=HTTPX_TIMEOUT, headers=HEADERS, follow_redirects=True
    ) as client:
        if "sbb" in scrapers_to_run:
            all_examples.extend(SBBScraper().collect(client, limit=per_scraper))
        if "zvv" in scrapers_to_run:
            all_examples.extend(ZVVScraper().collect(client, limit=per_scraper))
        if "admin" in scrapers_to_run:
            all_examples.extend(AdminCHScraper().collect(client, limit=per_scraper))
        if "news" in scrapers_to_run:
            all_examples.extend(NewsScraper().collect(client, limit=per_scraper))
        if "products" in scrapers_to_run:
            all_examples.extend(
                ProductCatalogScraper().collect(client, limit=per_scraper)
            )

    logger.info("Total examples collected: %d", len(all_examples))

    if not all_examples:
        logger.error("No examples collected. Check network access and scraper configs.")
        return

    builder = SwissDatasetBuilder(all_examples)
    output_dir = Path(args.output_dir) if args.output_dir else None
    ds = builder.build(output_dir)

    logger.info("Dataset splits: %s", {k: len(v) for k, v in ds.items()})
    logger.info("Sample train example: %s", ds["train"][0] if len(ds["train"]) > 0 else "N/A")

    # Print summary
    print("\n" + "=" * 60)
    print("SWISS MULTIMODAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Total examples: {len(all_examples)}")
    for split_name, split_ds in ds.items():
        print(f"  {split_name}: {len(split_ds)} examples")
    print(f"\nSources: {sorted(set(e.source for e in all_examples))}")
    print(f"Categories: {sorted(set(e.category for e in all_examples))}")
    print(f"Languages: {sorted(set(e.language for e in all_examples))}")
    print(f"\nDataset saved to: {output_dir or DATA_DIR / 'dataset'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
