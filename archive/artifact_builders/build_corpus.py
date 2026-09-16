import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


OUT = Path("work")
OUT.mkdir(exist_ok=True)

QUERIES = [
    "catastrophic forgetting large language model fine tuning",
    "continual learning large language models instruction tuning",
    "LoRA catastrophic forgetting",
    "parameter efficient fine tuning knowledge retention",
    "continual instruction tuning LLM replay",
    "spurious forgetting language models",
    "knowledge distillation forgetting LLM fine tuning",
    "spectral subspace LoRA forgetting",
    "gradient conflict continual fine tuning language model",
    "adapter composition interference continual learning LLM",
]

POS = {
    "catastrophic": 6, "forgetting": 6, "retention": 4, "continual": 4,
    "fine-tuning": 3, "finetuning": 3, "fine tuning": 3, "lora": 4,
    "parameter-efficient": 3, "parameter efficient": 3, "replay": 3,
    "rehearsal": 3, "distillation": 2, "subspace": 2, "spectral": 2,
    "instruction tuning": 2, "knowledge loss": 4, "capability degradation": 4,
    "alignment tax": 3, "gradient conflict": 3, "model merging": 1,
    "language model": 2, "llm": 2,
}
NEG = {
    "image classification": -5, "object detection": -5, "medical imaging": -5,
    "autonomous driving": -4, "robot": -3, "wireless": -4, "federated": -2,
    "recommendation": -2, "class-incremental": -2,
}


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "SOTA-review/1.0 (course-project-research)"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def inverted_to_text(inv):
    if not inv:
        return ""
    seq = []
    for tok, positions in inv.items():
        for p in positions:
            seq.append((p, tok))
    return " ".join(tok for _, tok in sorted(seq))


def norm_title(title):
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def score(title, abstract):
    text = f"{title} {abstract}".lower()
    s = sum(w for k, w in POS.items() if k in text)
    s += sum(w for k, w in NEG.items() if k in text)
    if "large language model" in text or "llm" in text:
        s += 3
    if "vision-language" in text or "multimodal large language" in text:
        s -= 1
    return s


rows = {}
raw_openalex = 0
for q in QUERIES:
    params = {
        "search": q,
        "filter": "from_publication_date:2023-01-01,to_publication_date:2026-09-04",
        "per-page": 100,
        "select": "id,doi,title,publication_year,publication_date,type,authorships,primary_location,abstract_inverted_index,cited_by_count,open_access",
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = get_json(url)
    raw_openalex += len(data.get("results", []))
    for w in data.get("results", []):
        title = w.get("title") or ""
        key = norm_title(title)
        if not key:
            continue
        abstract = inverted_to_text(w.get("abstract_inverted_index"))
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        authors = ", ".join(
            (a.get("author") or {}).get("display_name", "")
            for a in (w.get("authorships") or [])[:8]
        )
        url2 = w.get("doi") or loc.get("landing_page_url") or w.get("id") or ""
        rec = {
            "title": title,
            "authors": authors,
            "year": w.get("publication_year"),
            "date": w.get("publication_date"),
            "venue": src.get("display_name", ""),
            "type": w.get("type", ""),
            "url": url2,
            "doi": w.get("doi") or "",
            "citations": w.get("cited_by_count", 0),
            "abstract": abstract,
            "source_index": "OpenAlex",
        }
        rec["relevance_score"] = score(title, abstract)
        old = rows.get(key)
        if not old or rec["relevance_score"] > old["relevance_score"]:
            rows[key] = rec
    time.sleep(0.15)

# Add arXiv records to improve coverage of very recent 2026 work.
ns = {"a": "http://www.w3.org/2005/Atom"}
raw_arxiv = 0
for q in QUERIES[:8]:
    aq = f'all:"{q}"'
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": aq, "start": 0, "max_results": 80,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    req = urllib.request.Request(url, headers={"User-Agent": "SOTA-review/1.0 course project"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            root = ET.fromstring(r.read())
    except Exception:
        continue
    entries = root.findall("a:entry", ns)
    raw_arxiv += len(entries)
    for e in entries:
        title = " ".join((e.findtext("a:title", default="", namespaces=ns)).split())
        key = norm_title(title)
        if not key:
            continue
        abstract = " ".join((e.findtext("a:summary", default="", namespaces=ns)).split())
        date = e.findtext("a:published", default="", namespaces=ns)[:10]
        year = int(date[:4]) if date[:4].isdigit() else None
        if not year or year < 2023 or date > "2026-09-04":
            continue
        authors = ", ".join(a.findtext("a:name", default="", namespaces=ns) for a in e.findall("a:author", ns)[:8])
        aid = e.findtext("a:id", default="", namespaces=ns)
        rec = {
            "title": title, "authors": authors, "year": year, "date": date,
            "venue": "arXiv", "type": "preprint", "url": aid, "doi": "",
            "citations": 0, "abstract": abstract, "source_index": "arXiv",
            "relevance_score": score(title, abstract),
        }
        old = rows.get(key)
        if not old:
            rows[key] = rec
        elif old.get("venue", "") in ("", "arXiv") and rec["relevance_score"] >= old["relevance_score"]:
            rows[key] = rec
    time.sleep(0.5)

all_rows = sorted(rows.values(), key=lambda x: (x["relevance_score"], x["citations"], x["date"] or ""), reverse=True)

# The first-pass screen is deliberately bounded. High-scoring records are included;
# marginal records remain visible as excluded so the selection is auditable.
screened = [r for r in all_rows if r["relevance_score"] >= 10][:180]
for r in screened:
    decoder_signal = any(k in (r["title"] + " " + r["abstract"]).lower() for k in ["large language model", "llm", "lora", "instruction"])
    r["screen_decision"] = "Include" if r["relevance_score"] >= 15 and decoder_signal else "Exclude after abstract"
    if r["screen_decision"] == "Include":
        r["screen_note"] = "Directly addresses LLM fine-tuning/continual-learning retention or a transferable mechanism."
    elif not decoder_signal:
        r["screen_note"] = "Related continual-learning terminology but limited direct relevance to decoder-only LLM fine-tuning."
    else:
        r["screen_note"] = "Marginal mechanism/application; retained in audit trail but not close-read."

fields = ["title", "authors", "year", "date", "venue", "type", "screen_decision", "relevance_score", "citations", "screen_note", "url", "doi", "abstract", "source_index"]
with (OUT / "screened_corpus.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(screened)

stats = {
    "raw_openalex": raw_openalex,
    "raw_arxiv": raw_arxiv,
    "raw_total": raw_openalex + raw_arxiv,
    "unique_after_cross_source_dedup": len(rows),
    "screened_records": len(screened),
    "included_after_title_abstract": sum(r["screen_decision"] == "Include" for r in screened),
    "excluded_after_title_abstract": sum(r["screen_decision"] != "Include" for r in screened),
}
(OUT / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(json.dumps(stats, indent=2))
