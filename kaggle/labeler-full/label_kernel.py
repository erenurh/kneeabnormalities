"""LLM report labeler v2 (Qwen3-8B, offline, Kaggle T4x2).

For each report, emits a grade 0-4 per finding with per-finding severity
anchors (see SYSTEM). Grades are mapped to probabilities later, calibrated
on the gold-58.

SMOKE=True labels only the 58 gold studies; False labels everything.
Output: /kaggle/working/grades.csv (StudyInstanceUID + 12 grade columns + raw).
"""
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SMOKE = False
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

SYSTEM = """You are an expert musculoskeletal radiologist. Read the knee MRI report (any language: English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French, Bosnian) and grade 12 findings on a 0-4 scale:

0 = explicitly stated ABSENT / normal
1 = not mentioned at all
2 = mentioned, but MILD / low-grade / trace / degenerative-only
3 = moderate severity, OR probable/partial at the threshold boundary
4 = severe / large / definite full abnormality

Per-finding anchors (use the report's own wording):
- acl: 2=mucoid degeneration, sprain, low-grade partial tear; 3=high-grade partial tear (>50% fibers); 4=complete tear/discontinuity
- mcl: 2=low-grade sprain (grade 1), chronic thickening; 3=grade 2 / high-grade partial ACUTE tear; 4=complete (grade 3) tear
- med_men / lat_men: 2=intrasubstance/degenerative signal NOT reaching surface; 3=signal likely reaching surface, small/possible tear; 4=definite tear, complex/displaced/truncated/radial/root tear
- med_oa / lat_oa / pf_oa: 2=chondropathy grade 1-2 / mild cartilage thinning in that compartment; 3=grade 3, focal high-grade loss; 4=grade 4, full-thickness loss, bone-on-bone
- effusion: 2=trace/small/mild/minimal; 3=moderate; 4=large/severe
- synovitis: 2=mild/possible synovial thickening; 3=definite synovitis; 4=marked/severe
- bakers: 2=small cyst; 3=moderate; 4=large
- contusion: 2=subtle/small marrow edema; 3=definite bone bruise/contusion; 4=extensive marrow edema
- fracture: 2=old/healed/chronic or questionable; 3=probable acute fracture; 4=definite acute fracture line

Output ONLY a JSON object, no other text:
{"acl":g,"mcl":g,"med_men":g,"lat_men":g,"med_oa":g,"lat_oa":g,"pf_oa":g,"effusion":g,"synovitis":g,"bakers":g,"contusion":g,"fracture":g}"""


def parse(text):
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return [int(d[k]) if k in d and int(d[k]) in (0, 1, 2, 3, 4) else 1 for k in KEYS]
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
