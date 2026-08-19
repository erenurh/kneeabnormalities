"""LLM report labeler v1 (Qwen3-4B, offline, Kaggle GPU).

For each report, emits a grade 0-3 per finding:
  0 = explicitly negative/absent, 1 = not mentioned,
  2 = mentioned but mild/borderline/uncertain, 3 = clearly positive at threshold.
Grades are mapped to probabilities later, calibrated on the gold-58.

SMOKE=True labels only the 58 gold studies; False labels everything.
Output: /kaggle/working/grades.csv (StudyInstanceUID + 12 grade columns + raw).
"""
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SMOKE = True
BATCH = 8
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "med_men", "lat_men", "med_oa", "lat_oa", "pf_oa",
        "effusion", "synovitis", "bakers", "contusion", "fracture"]

INPUT = Path("/kaggle/input")
COMP = sorted(p.parent for p in INPUT.glob("*/train_series.csv")) or \
       sorted(p.parent for p in INPUT.glob("*/*/train_series.csv"))
MODEL = sorted(p.parent for p in INPUT.rglob("qwen*/**/config.json")) or \
        sorted(p.parent for p in INPUT.rglob("**/config.json"))
print("comp root:", COMP[0], "| model root:", MODEL[0])

SYSTEM = """You are an expert musculoskeletal radiologist. You will read a knee MRI radiology report (it may be in any language: English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French, Bosnian) and grade 12 findings.

For each finding output an integer grade:
0 = report explicitly states the finding is ABSENT/normal
1 = report does not mention the finding at all
2 = mentioned but MILD/borderline/low-grade/uncertain (below threshold)
3 = clearly POSITIVE at or above the threshold below

Thresholds (a finding is 3 only if it meets these; below them use 2):
- acl: high-grade partial or complete tear (discontinuity or >50% fibers). Mucoid degeneration/sprain/intact = 2 if mentioned abnormal.
- mcl: high-grade or complete ACUTE tear. Low-grade sprain/chronic thickening = 2.
- med_men / lat_men (medial/lateral meniscus): tear reaching the surface, or truncated/displaced/degenerated fragment. Intrasubstance signal only = 2.
- med_oa / lat_oa / pf_oa (medial/lateral tibiofemoral, patellofemoral compartment osteoarthritis): substantial cartilage loss (high-grade chondropathy, grade 3-4, bone-on-bone, osteophytes with cartilage loss) in that compartment. Mild chondropathy grade 1-2 = 2.
- effusion: moderate or large joint effusion. Trace/small/mild/minimal = 2.
- synovitis: synovial thickening/inflammation stated.
- bakers: moderate or large Baker's/popliteal cyst. Small cyst = 2.
- contusion: bone marrow edema/bruise WITHOUT fracture line.
- fracture: acute fracture line/cortical break.

Output ONLY a JSON object, no other text:
{"acl":g,"mcl":g,"med_men":g,"lat_men":g,"med_oa":g,"lat_oa":g,"pf_oa":g,"effusion":g,"synovitis":g,"bakers":g,"contusion":g,"fracture":g}"""


def parse(text):
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return [int(d[k]) if k in d and int(d[k]) in (0, 1, 2, 3) else 1 for k in KEYS]
    except (ValueError, KeyError, TypeError):
        return None


def main():
    tr = pd.read_csv(COMP[0] / "train.csv")
    if SMOKE:
        tr = tr[tr[LABELS].notna().all(axis=1)]
    print(len(tr), "reports to label")

    tok = AutoTokenizer.from_pretrained(MODEL[0], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL[0], dtype=torch.float16, device_map="auto")
    model.eval()

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": str(r)[:6000]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for r in tr["Report"].fillna("")]

    rows = []
    for i in range(0, len(prompts), BATCH):
        chunk = prompts[i:i + BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=3000).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=150, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        for j, o in enumerate(out):
            text = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            rows.append({"raw": text, "grades": parse(text)})
        if i % (BATCH * 10) == 0:
            print(i, flush=True)

    res = pd.DataFrame({"StudyInstanceUID": tr["StudyInstanceUID"].values})
    grades = [r["grades"] or [1] * 12 for r in rows]
    res[LABELS] = pd.DataFrame(grades, index=res.index)
    res["parse_ok"] = [r["grades"] is not None for r in rows]
    res["raw"] = [r["raw"][:300] for r in rows]
    res.to_csv("/kaggle/working/grades.csv", index=False)
    print("parse failures:", int((~res["parse_ok"]).sum()))


if __name__ == "__main__":
    main()
