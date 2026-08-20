"""LLM report labeler v3 (Qwen3-8B, thinking mode, offline, Kaggle T4x2).

For each report and finding, emits [grade 0-4, severity 0-100]:
grade uses per-finding anchors (SYSTEM); severity is the model's probability
(in %) that the finding is positive at the gold threshold. Grades/severities
are mapped to probabilities later, calibrated on the gold-58.

SMOKE=True labels only the 58 gold studies; False labels everything.
Output: /kaggle/working/grades.csv.
"""
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SMOKE = True
THINK = False
BATCH = 12
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

SYSTEM = """You are an expert musculoskeletal radiologist. Read the knee MRI report (any language: English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French, Bosnian) and assess 12 findings.

For each finding output [grade, severity]:

grade (integer 0-4):
0 = explicitly stated ABSENT / normal
1 = not mentioned at all
2 = mentioned, but MILD / low-grade / trace / degenerative-only
3 = moderate severity, OR probable/partial at the threshold boundary
4 = severe / large / definite full abnormality

Per-finding grade anchors (use the report's own wording):
- acl: 2=mucoid degeneration, sprain, low-grade partial tear; 3=high-grade partial tear (>50% fibers); 4=complete tear/discontinuity
- mcl: 2=low-grade sprain (grade 1) or chronic thickening WITHOUT surrounding edema; 3=grade 2 / high-grade partial ACUTE tear (fiber disruption or surrounding edema present); 4=complete (grade 3) tear
- med_men / lat_men: 2=intrasubstance/degenerative signal NOT reaching surface; 3=signal likely reaching surface, small/possible tear; 4=definite tear, complex/displaced/truncated/radial/root tear
- med_oa / lat_oa / pf_oa: 2=chondropathy grade 1-2 / mild cartilage thinning in that compartment; 3=grade 3, focal high-grade loss; 4=grade 4, full-thickness loss, bone-on-bone. Chondromalacia patellae / patellar or trochlear chondropathy belongs to pf_oa (its grade maps directly). Femorotibial medial→med_oa, lateral→lat_oa
- effusion: 2=trace/small/mild/minimal; 3=moderate; 4=large/severe
- synovitis: any-language synonyms count (synovial thickening/hypertrophy/proliferation, sinovit, sinovyal kalinlasma, hipertrofia sinovial, engrosamiento sinovial). 2=mild/possible; 3=definite; 4=marked/severe
- bakers: 2=small cyst; 3=moderate; 4=large
- contusion: 2=subtle/small marrow edema; 3=definite bone bruise/contusion; 4=extensive marrow edema
- fracture: 2=old/healed/chronic or questionable; 3=probable acute fracture; 4=definite acute fracture line

severity (integer 0-100): your probability in percent that the finding is POSITIVE at these strict thresholds on the images:
ACL/MCL positive only if high-grade partial or complete tear. Meniscus positive if tear reaches the surface. OA positive if >=1cm of >50%-thickness cartilage loss in that compartment. Effusion positive only if moderate or large. Baker's positive only if moderate or large. Contusion positive if impact marrow edema without fracture line. Fracture positive if acute fracture line. Borderline cases are NEGATIVE. A finding never mentioned may still be present: use a low but non-zero probability typical for knee MRI populations.

After thinking, output ONLY a JSON object:
{"acl":[g,s],"mcl":[g,s],"med_men":[g,s],"lat_men":[g,s],"med_oa":[g,s],"lat_oa":[g,s],"pf_oa":[g,s],"effusion":[g,s],"synovitis":[g,s],"bakers":[g,s],"contusion":[g,s],"fracture":[g,s]}"""


def parse(text):
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        out = []
        for k in KEYS:
            v = d[k]
            g, s = (v if isinstance(v, list) else [v, None])[:2]
            g = int(g)
            if g not in (0, 1, 2, 3, 4):
                return None
            s = min(100, max(0, int(s))) if s is not None else g * 25
            out.append((g, s))
        return out
    except (ValueError, KeyError, TypeError, IndexError):
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
        tokenize=False, add_generation_prompt=True, enable_thinking=THINK)
        for r in tr["Report"].fillna("")]

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    prompts = [prompts[i] for i in order]
    rows = [None] * len(prompts)
    done = 0
    for i in range(0, len(prompts), BATCH):
        chunk = prompts[i:i + BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=3000).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1600 if THINK else 200,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        for j, o in enumerate(out):
            text = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            rows[order[i + j]] = {"raw": text, "vals": parse(text)}
            done += 1
        if i % (BATCH * 5) == 0:
            print(done, flush=True)

    res = pd.DataFrame({"StudyInstanceUID": tr["StudyInstanceUID"].values})
    vals = [r["vals"] or [(1, 25)] * 12 for r in rows]
    res[LABELS] = pd.DataFrame([[g for g, _ in v] for v in vals], index=res.index)
    res[[c + "_sev" for c in LABELS]] = pd.DataFrame(
        [[s for _, s in v] for v in vals], index=res.index)
    res["parse_ok"] = [r["vals"] is not None for r in rows]
    res["raw"] = [r["raw"][-400:] for r in rows]
    res.to_csv("/kaggle/working/grades.csv", index=False)
    print("parse failures:", int((~res["parse_ok"]).sum()))


if __name__ == "__main__":
    main()
