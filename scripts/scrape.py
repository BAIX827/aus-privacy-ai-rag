"""Download OAIC privacy guidance (CC BY 4.0) into data/docs.jsonl, one record per section."""

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup

BASE = "https://www.oaic.gov.au"
GUIDANCE = BASE + "/privacy/privacy-guidance-for-organisations-and-government-agencies/"
APPG = BASE + "/privacy/australian-privacy-principles/australian-privacy-principles-guidelines/"
PAGES = {
    "ai-commercial-products": GUIDANCE
    + "guidance-on-privacy-and-the-use-of-commercially-available-ai-products",
    "ai-training-models": GUIDANCE + "guidance-on-privacy-and-developing-and-training-generative-ai-models",
    "app-a": APPG + "chapter-a-introductory-matters",
    "app-b": APPG + "chapter-b-key-concepts",
    "app-c": APPG + "chapter-c-permitted-general-situations",
    "app-d": APPG + "chapter-d-permitted-health-situations",
    **{
        f"app-{n}": APPG + f"chapter-{n}-app-{n}-{slug}"
        for n, slug in [
            (1, "open-and-transparent-management-of-personal-information"),
            (2, "anonymity-and-pseudonymity"),
            (3, "collection-of-solicited-personal-information"),
            (4, "dealing-with-unsolicited-personal-information"),
            (5, "notification-of-the-collection-of-personal-information"),
            (6, "use-or-disclosure-of-personal-information"),
            (7, "direct-marketing"),
            (8, "cross-border-disclosure-of-personal-information"),
            (9, "adoption,-use-or-disclosure-of-government-related-identifiers"),
            (10, "quality-of-personal-information"),
            (11, "security-of-personal-information"),
            (12, "access-to-personal-information"),
            (13, "correction-of-personal-information"),
        ]
    },
}
UA = "privacy-rag-research/0.1 (portfolio project; polite crawler)"
DELAY = 3.0
EMAIL = re.compile(r"\S+@\S+\.\w+")
PHONE = re.compile(r"\b(?:\+?61|0)[\d ]{8,12}\d\b")


def clean(text: str) -> str:
    return PHONE.sub("[phone]", EMAIL.sub("[email]", re.sub(r"\s+", " ", text))).strip()


def sections(html: str):
    main = BeautifulSoup(html, "lxml").find("main")
    for bad in main.select("nav, .breadcrumb__wrapper, .feedback, script, style"):
        bad.decompose()
    h1 = main.find("h1")
    title = h1.get_text(strip=True)
    path, buf = [title], []
    for el in h1.find_all_next(["h2", "h3", "h4", "p", "li", "td"]):
        if el.name in ("h2", "h3", "h4"):
            if buf:
                yield title, " > ".join(path[1:]) or "Introduction", clean(" ".join(buf))
            head = el.get_text(" ", strip=True)
            if head in ("On this page", "Downloads"):
                head = None
            level = int(el.name[1]) - 1
            path = path[:level] + ([head] if head else [])
            buf = []
        elif not el.find(["p", "li"]):  # leaf text only, avoids duplicates
            buf.append(el.get_text(" ", strip=True))
    if buf:
        yield title, " > ".join(path[1:]) or "Introduction", clean(" ".join(buf))


def disallowed(robots: str):
    """Disallow rules for '*' as regexes (supports * and $ wildcards, unlike urllib.robotparser)."""
    rules, applies = [], False
    for line in robots.splitlines():
        key, _, val = line.partition(":")
        key, val = key.strip().lower(), val.strip()
        if key == "user-agent":
            applies = val == "*"
        elif key == "disallow" and applies and val:
            rules.append(re.compile(re.escape(val).replace(r"\*", ".*").replace(r"\$", "$")))
    return rules


def main(out="data/docs.jsonl"):
    rows = []
    with httpx.Client(headers={"User-Agent": UA}, timeout=30, follow_redirects=True) as client:
        rules = disallowed(client.get(BASE + "/robots.txt").text)
        for doc_id, url in PAGES.items():
            parts = urlsplit(url)
            if any(r.match(parts.path + ("?" + parts.query if parts.query else "")) for r in rules):
                print("robots.txt disallows", url, file=sys.stderr)
                continue
            html = client.get(url).raise_for_status().text
            for title, section, text in sections(html):
                if len(text) > 40 and "Footnotes" not in section:
                    rows.append(
                        {"doc_id": doc_id, "title": title, "section": section, "url": url, "text": text}
                    )
            print(doc_id, len(rows), file=sys.stderr)
            time.sleep(DELAY)
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
