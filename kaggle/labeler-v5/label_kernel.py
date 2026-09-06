"""LLM report labeler v5 (Qwen3-8B, offline, Kaggle T4x2) — WEAK-FOUR ONLY.

Targets the four findings that are teacher-limited or model-limited at the
LB-0.922 checkpoint (gold-58: Synovitis .839, Lateral OA .845, PF OA .876,
Lateral Meniscus .872) and where our own v3 labeler is weakest (.75-.82).
Differences from v3: only 4 findings (short output, ~3x faster than the 5.6 h
v3 full run), per-finding English evidence quote (translate-then-label inside
the answer, hand-checkable), much richer multilingual anchors, and explicit
laterality / compartment disambiguation rules.

Output per finding [grade 0-4, severity 0-100, evidence]. Grades are mapped to
probabilities later on the gold-58 (src/calibrate_grades.py); severities are
used raw. SMOKE=True labels only the 58 gold studies (gate run).
Output: /kaggle/working/grades_v5.csv.
"""
import json
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SMOKE = True
BATCH = 8
MAX_NEW = 260
LABELS = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis"]
KEYS = ["lat_men", "lat_oa", "pf_oa", "synovitis"]
ALL_LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
              "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
              "Contusion", "Fracture"]

INPUT = Path("/kaggle/input")
COMP = sorted(p.parent for p in INPUT.glob("*/train_series.csv")) or \
       sorted(p.parent for p in INPUT.glob("*/*/train_series.csv"))
MODEL = sorted(p.parent for p in INPUT.rglob("qwen*/**/config.json")) or \
        sorted(p.parent for p in INPUT.rglob("**/config.json"))
print("comp root:", COMP[0], "| model root:", MODEL[0])

SYSTEM = """You are an expert musculoskeletal radiologist. Read the knee MRI report (any language: English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French, Bosnian) and assess exactly FOUR findings. Ignore everything else in the report.

For each finding output [grade, severity, evidence]:

grade (integer 0-4):
0 = explicitly stated ABSENT / normal / intact for THIS structure or compartment
1 = not mentioned at all (a generic "no other abnormality" phrase still counts as 1, not 0)
2 = mentioned, but MILD / low-grade / degenerative-only / possible
3 = moderate, OR probable / at the threshold boundary
4 = severe / large / definite full abnormality

Anchors:

lat_men = LATERAL meniscus only (lateral, lateralni, dış/lateral menisküs, menisco externo/lateral, Außenmeniskus, έξω μηνίσκος, латерален менискус). Never use medial-meniscus sentences. Discoid lateral meniscus without tear = 2.
 2 = intrasubstance / degenerative / grade 1-2 signal NOT reaching an articular surface; "degeneration"; "mucoid"
 3 = signal probably reaching the surface, "possible / suspected tear", small horizontal tear of the posterior horn, "grade 3 signal" without confirmation
 4 = definite tear: reaches the surface, complex, radial, flap, bucket-handle, displaced, truncated, root tear, macerated, prior partial meniscectomy with re-tear

lat_oa = cartilage of the LATERAL femorotibial compartment only (lateral femoral condyle / lateral tibial plateau; lateral compartment; lateralni odeljak; lateral kompartman; compartimento externo/lateral; laterales Kompartiment; έξω διαμέρισμα). Never count medial or patellofemoral cartilage here.
 2 = chondropathy / chondromalacia grade 1-2, mild thinning, superficial fissures, "mild degenerative change", small osteophytes only
 3 = grade 3, focal deep / high-grade partial-thickness defect, "moderate" cartilage loss, subchondral cyst or edema attributed to cartilage loss
 4 = grade 4, full-thickness loss, bone exposed / bone-on-bone, "severe / advanced OA" of that compartment

pf_oa = PATELLOFEMORAL cartilage: patella (retropatellar) and/or trochlea (patellofemoral joint, femoropatelar, patellofemoralni, kondromalasi patella, condromalacia rotuliana, Retropatellararthrose, χονδροπάθεια επιγονατίδας, хондромалация на пателата). Chondromalacia patellae ALWAYS maps here, its grade maps directly.
 2 = chondromalacia / chondropathy grade 1-2, softening, surface irregularity, mild thinning, "mild patellofemoral degenerative change"
 3 = grade 3, deep fissuring or partial-thickness defect >50%, "moderate"
 4 = grade 4, full-thickness defect with exposed bone, "severe / advanced patellofemoral OA"

synovitis = inflamed / thickened / proliferative synovium anywhere in the knee. Synonyms: synovial thickening / hypertrophy / proliferation / enhancement / irregularity, sinovitis, sinovit, sinovyal kalınlaşma / hipertrofi, hipertrofia / engrosamiento sinovial, Synovialitis, Synoviaverdickung, υμενίτιδα, синовит, synoviale verdikking, plica synovialis with inflammation, villonodular synovitis, lipoma arborescens, frond-like synovium, synovial debris / loose bodies with thickening, Hoffa fat-pad synovitis. A plain joint effusion WITHOUT any synovial descriptor is NOT synovitis (leave synovitis at 1).
 2 = mild / minimal / possible / "some synovial thickening"
 3 = definite / moderate synovitis, synovial hypertrophy
 4 = marked / severe / diffuse / villonodular / extensive proliferation

severity (integer 0-100): your probability in percent that the finding is POSITIVE at these strict image-based thresholds: lat_men positive only if a tear reaches the articular surface; lat_oa and pf_oa positive only if >=1 cm of >50%-thickness cartilage loss in that compartment; synovitis positive if synovial thickening / proliferation is visible. Borderline is NEGATIVE. A finding never mentioned may still be present; give a low but non-zero probability typical for symptomatic knee MRI patients (roughly: lat_men 10, lat_oa 8, pf_oa 15, synovitis 25).

evidence: the single most relevant report phrase for that finding, translated to English, max 12 words, or "" if not mentioned.

Output ONLY a JSON object, no prose:
{"lat_men":[g,s,"evidence"],"lat_oa":[g,s,"evidence"],"pf_oa":[g,s,"evidence"],"synovitis":[g,s,"evidence"]}"""


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
            if not isinstance(v, list):
                v = [v, None, ""]
            v = list(v) + [None, ""][len(v) - 1:] if len(v) < 3 else v
            g = int(v[0])
            if g not in (0, 1, 2, 3, 4):
                return None
            s = min(100, max(0, int(v[1]))) if v[1] is not None else g * 25
            out.append((g, s, str(v[2])[:120]))
        return out
    except (ValueError, KeyError, TypeError, IndexError):
        return None


def main():
    t0 = time.time()
    tr = pd.read_csv(COMP[0] / "train.csv")
    if SMOKE:
        tr = tr[tr[ALL_LABELS].notna().all(axis=1)]
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

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    prompts = [prompts[i] for i in order]
    rows = [None] * len(prompts)
    done = 0
    for i in range(0, len(prompts), BATCH):
        chunk = prompts[i:i + BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=3000).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        for j, o in enumerate(out):
            text = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            rows[order[i + j]] = {"raw": text, "vals": parse(text)}
            done += 1
        if i % (BATCH * 5) == 0:
            el = time.time() - t0
            print(f"{done} elapsed={el:.0f}s per_report={el/done:.2f}s", flush=True)

    res = pd.DataFrame({"StudyInstanceUID": tr["StudyInstanceUID"].values})
    vals = [r["vals"] or [(1, 25, "")] * 4 for r in rows]
    res[LABELS] = pd.DataFrame([[g for g, _, _ in v] for v in vals], index=res.index)
    res[[c + "_sev" for c in LABELS]] = pd.DataFrame(
        [[s for _, s, _ in v] for v in vals], index=res.index)
    res[[c + "_ev" for c in LABELS]] = pd.DataFrame(
        [[e for _, _, e in v] for v in vals], index=res.index)
    res["parse_ok"] = [r["vals"] is not None for r in rows]
    res["raw"] = [r["raw"][-300:] for r in rows]
    res.to_csv("/kaggle/working/grades_v5.csv", index=False)
    print("parse failures:", int((~res["parse_ok"]).sum()),
          f"total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
