"""
VeriBuy backend — real product-page fetch + Groq-powered reasoning +
Breeth-powered user-preference memory.

What this actually does (and does not do):
  - Fetches the real HTML of the URL you submit (httpx).
  - Extracts genuine data if the page exposes it: schema.org/Product JSON-LD,
    Open Graph tags, or basic meta fallbacks (title/price/rating/review count).
  - Sends ONLY that extracted data (plus any relevant remembered user
    preferences from Breeth) to Groq and asks it to reason about
    plausibility/risk -- it is explicitly instructed not to invent numbers
    for anything that wasn't actually found on the page.
  - After each analysis, writes a short memory of what was analyzed to
    Breeth so future requests from the same user can be informed by their
    past behavior/preferences (e.g. "tends to flag high-price electronics
    as risky", "usually buys from Brand X").
  - Returns a dataQuality object so the frontend can be honest about what
    was verified vs. unknown, instead of pretending every field is measured.

What this does NOT do:
  - It does not call any Amazon/Flipkart/etc. price API (none is configured).
  - It does not read individual customer reviews (most sites hide these
    behind JS rendering or bot protection) -- only the aggregate rating/count
    if the page publishes one.
  - It will honestly fail/flag low-confidence for sites that block scraping.
  - Breeth is used ONLY for remembering/recalling user preferences. It is
    NOT an LLM and does not generate the risk analysis itself.

Env vars (backend/.env):
  GROQ_API_KEY     - required for /analyze to work
  GROQ_MODEL       - optional, defaults to "llama-3.3-70b-versatile"
  BREETH_API_KEY   - optional; if unset, memory features are silently
                     skipped (analysis still works, just without personalization)
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Groq (replaces OpenAI) — does the actual authenticity/risk reasoning
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_CONFIGURED = bool(GROQ_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_CONFIGURED else None
if not GROQ_CONFIGURED:
    print(
        "[VeriBuy] WARNING: GROQ_API_KEY not set — the site will run, but "
        "live analysis will be unavailable and every request will fall back "
        "to simulated demo mode. Add a key to backend/.env to enable it."
    )

# ---------------------------------------------------------------------------
# Breeth — remembers user preferences/history across requests (NOT an LLM,
# just persistent memory: write episodes, search them back later)
# ---------------------------------------------------------------------------
BREETH_API_KEY = os.getenv("BREETH_API_KEY")
BREETH_BASE_URL = os.getenv("BREETH_BASE_URL", "https://api.thebreeth.com/v1")
BREETH_CONFIGURED = bool(BREETH_API_KEY)

if not BREETH_CONFIGURED:
    print(
        "[VeriBuy] NOTE: BREETH_API_KEY not set — analysis will still work, "
        "but responses won't be personalized with remembered user "
        "preferences. Add a key to backend/.env to enable it."
    )

app = FastAPI(title="Shop Now Backend")

# Frontend is served by this same app (see StaticFiles mount at the bottom),
# but CORS stays open in case the HTML is ever opened directly (file://) too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}


class AnalyzeRequest(BaseModel):
    url: str
    # Optional caller-supplied identifier so memories are kept per-user
    # instead of all landing in one shared bucket. Falls back to "default"
    # (single shared memory group) if not provided.
    user_id: Optional[str] = None


def extract_jsonld_product(soup: BeautifulSoup) -> Optional[dict]:
    """Look for a schema.org Product block. Many storefronts (Shopify,
    WooCommerce, some marketplace seller pages) include this; big
    marketplaces often strip it or render it via JS, in which case this
    returns None and we fall back to weaker signals."""
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if graph:
                candidates.extend(g for g in graph if isinstance(g, dict))
            item_type = item.get("@type")
            types = item_type if isinstance(item_type, list) else [item_type]
            if types and any(t and "product" in str(t).lower() for t in types):
                return item
    return None


def extract_meta_fallback(soup: BeautifulSoup) -> dict:
    def meta(name_or_prop):
        tag = soup.find("meta", attrs={"property": name_or_prop}) or soup.find(
            "meta", attrs={"name": name_or_prop}
        )
        return tag["content"].strip() if tag and tag.get("content") else None

    title = meta("og:title") or (soup.title.string.strip() if soup.title and soup.title.string else None)
    image = meta("og:image")
    price = meta("product:price:amount") or meta("og:price:amount")
    currency = meta("product:price:currency") or meta("og:price:currency")

    if not price:
        text = soup.get_text(" ", strip=True)
        m = re.search(r"(\u20b9|\$|\u00a3|\u20ac)\s?([\d,]{2,10}(?:\.\d{1,2})?)", text)
        if m:
            currency = currency or m.group(1)
            price = m.group(2).replace(",", "")

    return {"name": title, "image": image, "price": price, "currency": currency}


def extract_product_data(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    product = extract_jsonld_product(soup)
    fallback = extract_meta_fallback(soup)

    result = {
        "name": None,
        "brand": None,
        "image": None,
        "price": None,
        "currency": None,
        "rating": None,
        "reviewCount": None,
        "source": "none",
    }

    if product:
        result["source"] = "json-ld"
        result["name"] = product.get("name")
        brand = product.get("brand")
        if isinstance(brand, dict):
            result["brand"] = brand.get("name")
        elif isinstance(brand, str):
            result["brand"] = brand
        offers = product.get("offers")
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        if isinstance(offers, dict):
            result["price"] = offers.get("price") or offers.get("lowPrice")
            result["currency"] = offers.get("priceCurrency")
        rating = product.get("aggregateRating")
        if isinstance(rating, dict):
            result["rating"] = rating.get("ratingValue")
            result["reviewCount"] = rating.get("reviewCount") or rating.get("ratingCount")
        result["image"] = product.get("image")
        if isinstance(result["image"], list):
            result["image"] = result["image"][0] if result["image"] else None

    for key in ("name", "image", "price", "currency"):
        if not result.get(key) and fallback.get(key):
            result[key] = fallback[key]
            if result["source"] == "none":
                result["source"] = "meta-tags"

    return result


ANALYSIS_SYSTEM_PROMPT = """You are a cautious product-verification analyst for VeriBuy.
You will be given ONLY the data actually extracted from a live product page,
plus (optionally) a short list of remembered facts about this user's past
preferences/behavior -- never invent facts, review contents, or numbers that
were not provided. The remembered facts are context only; do not treat them
as ground truth about THIS product, only about the user's general tendencies.

Given the extracted data, respond with ONLY a JSON object with these fields:
- authenticity (integer 0-100, your best-effort estimate of genuine-product
  probability based on brand/price/name plausibility; if there is not enough
  data to judge, return null)
- authenticityReasoning (1-2 sentences, plain language)
- priceScore (integer 0-100 estimating whether the price looks plausible for
  this kind of product; null if no price was extracted)
- sellerTrust (null -- you have no seller data, always return null for this
  unless explicitly told otherwise)
- reviewTrust (integer 0-100 ONLY if a rating and review count were provided;
  otherwise null -- do not estimate this without real review data)
- overallRisk ("Low", "Medium", "High", or "Unknown" if not enough data)
- recommendation ("Safe to Buy", "Buy with Caution", "High Risk Purchase", or
  "Not Enough Data")
- explanation (2-4 sentences summarizing the reasoning, explicitly noting
  which parts of the analysis are based on real page data vs. not available)
- personalizationNote (1 sentence noting how, if at all, remembered user
  preferences influenced the read -- or null if none were supplied/used)

Return raw JSON only, no markdown fences, no extra text.
"""


def _extract_json_object(text: str) -> dict:
    """Groq/open-weight models don't always honor response_format as
    strictly as OpenAI does, so defensively strip code fences and pull out
    the first {...} block before parsing."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


async def breeth_search(query: str, group_id: str, limit: int = 5) -> list[dict]:
    """Pull back relevant remembered facts about this user. Returns an
    empty list (never raises to the caller) if Breeth isn't configured or
    the call fails -- personalization is a nice-to-have, not a dependency."""
    if not BREETH_CONFIGURED:
        return []
    try:
        async with httpx.AsyncClient(timeout=8.0) as http_client:
            resp = await http_client.post(
                f"{BREETH_BASE_URL}/search",
                headers={"Authorization": f"Bearer {BREETH_API_KEY}"},
                json={"query": query, "group_id": group_id, "limit": limit},
            )
        if resp.status_code != 200:
            return []
        return resp.json().get("edges", [])
    except httpx.RequestError:
        return []


async def breeth_remember(content: str, group_id: str) -> None:
    """Fire-and-forget write of a new memory. Failures are swallowed --
    losing a memory write should never break the actual analysis response."""
    if not BREETH_CONFIGURED:
        return
    try:
        async with httpx.AsyncClient(timeout=8.0) as http_client:
            await http_client.post(
                f"{BREETH_BASE_URL}/episodes",
                headers={"Authorization": f"Bearer {BREETH_API_KEY}"},
                json={"content": content, "group_id": group_id, "extract_intent": True},
            )
    except httpx.RequestError:
        pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "groq_configured": GROQ_CONFIGURED,
        "breeth_configured": BREETH_CONFIGURED,
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not GROQ_CONFIGURED:
        raise HTTPException(
            status_code=503,
            detail="No Groq API key configured on the server. Add one to backend/.env and restart uvicorn.",
        )

    url = req.url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    group_id = req.user_id or "default"

    try:
        async with httpx.AsyncClient(
            headers=BROWSER_HEADERS, follow_redirects=True, timeout=15.0
        ) as http_client:
            resp = await http_client.get(url)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach that URL: {e}")

    if resp.status_code in (403, 999):
        raise HTTPException(
            status_code=403,
            detail=(
                "This site blocked the request (common for Amazon and other "
                "large marketplaces that detect automated access). A real "
                "integration for this site would need an official partner "
                "API or a paid scraping/proxy service."
            ),
        )
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"The page returned HTTP {resp.status_code}.",
        )

    extracted = extract_product_data(resp.text)

    data_quality = {
        "extractionSource": extracted["source"],
        "hasPrice": extracted["price"] is not None,
        "hasRating": extracted["rating"] is not None,
        "hasReviewCount": extracted["reviewCount"] is not None,
        "note": (
            "No structured product data was found on this page -- the site "
            "likely renders content with JavaScript or blocks scrapers. "
            "Analysis below will be low-confidence."
            if extracted["source"] == "none"
            else None
        ),
    }

    # Pull back anything Breeth remembers about this user's buying
    # preferences/history that's relevant to this kind of product.
    query_bits = " ".join(
        filter(None, [extracted.get("brand"), extracted.get("name")])
    ) or "this product"
    remembered_edges = await breeth_search(
        f"What are this user's preferences or past behavior relevant to buying {query_bits}?",
        group_id=group_id,
    )
    remembered_facts = [e.get("fact") for e in remembered_edges if e.get("fact")]

    user_message = {
        "extractedData": extracted,
        "url": url,
        "rememberedUserFacts": remembered_facts,  # [] if none / Breeth not configured
    }

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_message)},
            ],
            temperature=0.3,
        )
        raw_content = completion.choices[0].message.content
        analysis = _extract_json_object(raw_content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq analysis failed: {e}")

    # Remember this analysis for next time (best-effort, non-blocking on
    # failure). Keep it factual/short -- this is what future searches will
    # retrieve as "rememberedUserFacts".
    summary_bits = [f"User analyzed a product page ({url})."]
    if extracted.get("name"):
        summary_bits.append(f"Product: {extracted['name']}.")
    if extracted.get("brand"):
        summary_bits.append(f"Brand: {extracted['brand']}.")
    if extracted.get("price"):
        summary_bits.append(f"Price: {extracted.get('currency') or ''}{extracted['price']}.")
    if analysis.get("recommendation"):
        summary_bits.append(f"VeriBuy recommendation was: {analysis['recommendation']}.")
    await breeth_remember(" ".join(summary_bits), group_id=group_id)

    return {
        "url": url,
        "extracted": extracted,
        "dataQuality": data_quality,
        "analysis": analysis,
        "personalization": {
            "breethConfigured": BREETH_CONFIGURED,
            "rememberedFactsUsed": remembered_facts,
        },
    }


# ---------------------------------------------------------------------------
# Serve the frontend from this same server, so the whole site is just:
#   uvicorn main:app --port 8000
# and open http://localhost:8000
# Mounted LAST and at "/" so it never shadows the /health and /analyze
# routes above (Starlette matches routes in registration order).
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    print(f"[VeriBuy] WARNING: frontend directory not found at {FRONTEND_DIR}")
