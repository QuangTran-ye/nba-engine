"""Parser coverage check: every status word in a report becomes exactly one parsed row.
Run: .venv/bin/python tests/test_injury_parser.py   (needs cached PDFs in data/raw/injury_reports)"""
import random, re, sys
sys.path.insert(0, ".")
import pdfplumber
from nba.config import RAW
from nba.injuries import STATUSES, norm, parse_pdf, report_key

for raw, full in [("BullockJr.,Reggie", "Reggie Bullock Jr."), ("Brown Jr., Troy", "Troy Brown Jr."),
                  ("Udeh,Jr.,Ernest", "Ernest Udeh Jr."), ("López,Karim", "Karim López"), ("LivelyII,Dereck", "Dereck Lively II")]:
    assert report_key(raw) == norm(full), (raw, report_key(raw), norm(full))

st = re.compile(r"\b(" + "|".join(STATUSES) + r")\b")
files = sorted(f for f in (RAW / "injury_reports").glob("*.pdf") if f.stat().st_size > 1000)
random.seed(0)
for f in random.sample(files, min(12, len(files))):
    with pdfplumber.open(f) as p:
        n = len(st.findall("\n".join(pg.extract_text() or "" for pg in p.pages)))
    got = len(parse_pdf(f))
    assert got == n, f"{f.name}: {got} parsed vs {n} status words"
print("injury parser ok")
