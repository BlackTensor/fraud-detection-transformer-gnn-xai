"""
explain_ollama.py — SHAP factors -> local Ollama natural-language explanation (Task 6.2).

What it does:
    Turns the numeric SHAP attribution from Task 6.1 into a short, plain-English
    paragraph a bank officer could read, using a LOCAL Ollama model (free, no API
    key — honours the project's ZERO-COST rule). It builds a prompt containing
    the predicted fraud probability, the raw transaction details, and
    the TOP-5 signed SHAP contributions, and it INSTRUCTS the model to explain
    ONLY from those numbers and to invent nothing.

Why this is "grounded":
    The model is told to use only the provided evidence. Because the prompt carries
    the actual SHAP signs/magnitudes and the raw fields, the generated sentences
    reference the real drivers (e.g. the C-count factors and device type that pushed
    this transaction's risk up) rather than hallucinated reasons.

Inputs:
    results/shap_example.json   written by explain_shap.py (prob + raw + ranked rows).
                                If absent, this script re-runs explain_shap.explain().
Model:
    Default 'llama3.2:3b'. 'phi3' was also evaluated, but on this
    project's prompts the tiny phi3 hallucinated MORE (it invented timing/region/
    device claims and mis-read feature values as percentages), so llama3.2:3b — which
    stayed closer to the listed evidence — is the chosen default. Both are free, local,
    and need no API key. If the requested model isn't installed we fall back to
    whatever IS installed so the demo still runs. (Pull models via `ollama pull <name>`.)

Prompt hardening:
    Because small local models tend to embellish, `build_prompt` adds STRICT rules
    on top of the base template: use only the given numbers, treat the C-codes as
    anonymized (don't guess their meaning), don't read the scaled "value=" as a
    percentage, and invent no timing/region/device/merchant claims.

Outputs:
    Prints the grounded paragraph; also saves it to results/explanation_example.md.

Run:  python src/explain_ollama.py
      python src/explain_ollama.py --model phi3 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

SHAP_JSON = config.RESULTS_DIR / "shap_example.json"
OUT_MD = config.RESULTS_DIR / "explanation_example.md"
DEFAULT_MODEL = "llama3.2:3b"


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def format_shap_lines(rows: list[dict], top_k: int = 5) -> str:
    """Render the top-k signed SHAP contributions as one line each.

    rows are already ranked by |contribution| (from explain_shap). Each line shows
    the feature, its model-input value, and the signed impact with a +/- gloss.
    """
    lines = []
    for r in rows[:top_k]:
        # State the direction in words right beside the signed SHAP number so the
        # model echoes the correct verb. A small model otherwise occasionally wrote
        # "raised risk by -0.070" for a risk-LOWERING factor (number right, verb
        # wrong). The scaled model input is kept as a bracketed aside — the prompt
        # rules still instruct the model not to read it as a percentage.
        direction = "raises risk" if r["shap_contribution"] > 0 else "lowers risk"
        lines.append(
            f"  - factor {r['feature']}: {r['shap_contribution']:+.3f} ({direction})"
            f"  [model input value={r['model_value']:.3f}]"
        )
    return "\n".join(lines)


def build_prompt(payload: dict, top_k: int = 5) -> str:
    """Build the grounded fraud-explanation prompt from the SHAP payload."""
    prob = float(payload["meta"]["predicted_prob"])
    raw = payload.get("raw", {})
    rows = payload["rows"]

    # Direction-aware closing instruction. The net SHAP push (vs the background base
    # rate) decides whether risk-RAISING or risk-LOWERING factors dominate; hard-coding
    # "risk-raising outweighed" would mis-frame the low-probability (legit) cases.
    base = payload["meta"].get("base_value")
    net_raise = (prob >= base) if base is not None else (prob >= 0.5)
    if net_raise:
        closing = ("Note that the risk-RAISING factors outweighed the rest, so the "
                   "transaction looks high-risk.")
    else:
        closing = ("Note that the risk-LOWERING factors outweighed the rest, so the "
                   "transaction looks low-risk despite any factors that raise risk.")

    amt = raw.get("TransactionAmt", "unknown")
    hour = raw.get("hour", "unknown")
    merchant = raw.get("ProductCD", "unknown")
    # Device: combine the coarse type with the device string when available.
    # cleaning step imputes missing identity fields to the literal "unknown",
    # so treat that placeholder (and blanks/NaN) as ABSENT — otherwise the prompt
    # reads a useless "unknown (unknown)" for the ~76% of rows with no identity row.
    def _present(v):
        if v is None:
            return None
        s = str(v).strip()
        return None if s == "" or s.lower() in ("unknown", "nan") else s

    dtype = _present(raw.get("DeviceType"))
    dinfo = _present(raw.get("DeviceInfo"))
    if dtype and dinfo:
        device = f"{dtype} ({dinfo})"
    else:
        device = dtype or dinfo or "unknown"
    region = raw.get("addr1", "unknown")

    ranked = format_shap_lines(rows, top_k=top_k)

    # The base template (Prediction / Transaction / Top factors) PLUS a hardened
    # rule block — small local models otherwise embellish (invent timing/region/device
    # claims, or read the scaled "value=" as a percentage). The rules keep the output
    # faithful to the listed SHAP evidence.
    return (
        "You are a fraud-analysis assistant. Explain the model's decision for a bank\n"
        "officer using ONLY the evidence listed below.\n\n"
        "STRICT RULES — follow all of them:\n"
        "- Use ONLY the numbers and fields given here. Do NOT add outside knowledge,\n"
        "  history, statistics, or any context that is not literally listed.\n"
        "- The factor codes (e.g. C1, C8, C11) are ANONYMIZED features. Do NOT guess,\n"
        "  state, or imply what they represent.\n"
        "- Refer to each factor ONLY by its exact code/name as written. NEVER describe a\n"
        "  C... or D... factor as a device, email, merchant, region, time, OS, or browser\n"
        "  feature — those codes are anonymized and unnamed.\n"
        "- Each factor's 'value=' is the model's internal scaled input, NOT a\n"
        "  probability or percentage — do not convert it to a percent or reinterpret it.\n"
        "- Describe each factor only as raising or lowering risk by its SHAP number.\n"
        "- Do NOT invent claims about timing, location, device reputation, or merchant\n"
        "  behaviour beyond what is explicitly written.\n\n"
        f"Prediction: {prob:.0%} fraud\n"
        f"Transaction: amount={amt}, hour={hour}, merchant={merchant},\n"
        f"             device={device}, region={region}\n"
        "Top contributing factors (SHAP, + raises risk, - lowers it):\n"
        f"{ranked}\n\n"
        "Write 3-4 plain-English sentences a bank officer could read. Refer to each\n"
        "factor as 'factor <CODE>' using its EXACT code and its EXACT signed SHAP value\n"
        "from the list above (never the 'value=' number) — e.g. phrased like \"factor Z9\n"
        f"raised risk by +0.21\" (substitute the real codes and their real SHAP values).\n"
        f"{closing}"
    )


# --------------------------------------------------------------------------- #
# Ollama model selection + call (local, free)
# --------------------------------------------------------------------------- #
def installed_models() -> list[str]:
    """Return the names of locally-installed Ollama models (robust to API shape)."""
    import ollama

    resp = ollama.list()
    models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
    names = []
    for m in models:
        if isinstance(m, dict):
            names.append(m.get("model") or m.get("name"))
        else:
            names.append(getattr(m, "model", None) or getattr(m, "name", None))
    return [n for n in names if n]


def pick_model(preferred: str = DEFAULT_MODEL) -> str:
    """Choose the model to use: `preferred` if installed, else the first installed.

    Raises a clear, actionable error if Ollama has no models at all.
    """
    names = installed_models()
    if not names:
        raise RuntimeError(
            "Ollama is running but has no models installed. Pull a free one first, "
            "e.g.  `ollama pull phi3`  (or `ollama pull mistral`)."
        )
    # Exact match, or match ignoring the :tag (so 'phi3' matches 'phi3:latest').
    base = preferred.split(":")[0]
    for n in names:
        if n == preferred or n.split(":")[0] == base:
            return n
    fallback = names[0]
    print(f"[ollama] preferred model '{preferred}' not installed; "
          f"falling back to '{fallback}'. (Installed: {', '.join(names)})")
    return fallback


def strip_meta_commentary(text: str) -> str:
    """Remove small-model chattiness from an LLM explanation, keeping only the
    officer-facing text.

    Two kinds of noise a 3B model tends to add: a chatty PREAMBLE before the
    explanation ("Sure, here's a summary:") and a TRAILING self-note about its own
    process or the prompt format ("Note: I've followed the exact format requested in
    the prompt"). Both are stripped. The stripping is deliberately CONSERVATIVE — it
    keys on process/format wording, so it never removes a legitimate
    "Note that the risk-RAISING factors outweighed ..." sentence that the prompt
    itself asks the model to write.
    """
    import re

    # Wording that marks a sentence/line as meta-commentary (not fraud evidence).
    meta = re.compile(
        r"(followed (the )?(exact )?format"
        r"|(requested|specified|provided|given|desired) format"
        r"|format (you )?(requested|specified|provided|asked|wanted)"
        r"|as (per |you )?requested"
        r"|i hope (this|that|it) (helps|is helpful)"
        r"|let me know if"
        r"|feel free to"
        r"|i('ve| have)? ?(kept|stuck to|adhered to|used|followed|maintained) the"
        r"|this (response|explanation|summary|answer) (follows|adheres|meets|uses|is in)"
        r"|(per|using|in line with|as per) (the|your) (prompt|instructions?|requested format)"
        r"|word (count|limit)"
        r"|\d+[- ]sentence"
        # Self-referential notes about obeying the prompt (e.g. "I've only referred to
        # each factor by its exact code ... as per your instructions"). These are
        # process commentary, never officer-facing evidence.
        r"|as (per )?(your |the )?instruct(ed|ions)"
        r"|as instructed"
        r"|your instructions"
        r"|referr?(ed|ing|) to each (factor|code)"
        r"|by (its|their) exact (code|name)"
        r"|exact code(s)? and (signed|its)"
        r"|using (its|their|the) exact code"
        r"|as (you )?(asked|specified|wanted|instructed|stated)"
        # Openers that echo the prompt's own format instruction (e.g. "Here are 3-4
        # plain-English sentences a bank officer could read:").
        r"|bank officer could read"
        r"|plain[- ]?english sentence"
        r"|\d+\s*(to|-|–)?\s*\d*\s*(plain[- ]?english )?sentences?)",
        re.IGNORECASE,
    )
    # A short opener segment that just announces the explanation.
    preamble = re.compile(
        r"^\s*(sure|certainly|of course|absolutely|here'?s|here is|here are|below is|"
        r"the following)\b",
        re.IGNORECASE,
    )

    # Remove a leading opener phrase that echoes the prompt and ends in a colon
    # (e.g. "Here are 3-4 plain-English sentences a bank officer could read:"),
    # keeping the real explanation that follows on the same line. The [^:.!?]* stops
    # at the first colon and never crosses a sentence boundary, so only the opener is
    # removed — not the content sentence the model fuses onto it after the colon.
    text = re.sub(
        r"^\s*(sure|certainly|of course|absolutely|here'?s|here is|here are|"
        r"below is|the following)\b[^:.!?]*:\s*",
        "", text.strip(), flags=re.IGNORECASE,
    )

    # Split into segments on sentence boundaries AND newlines, so a meta sentence
    # fused onto a content sentence is separated from it (and standalone "Note:"
    # lines without trailing punctuation are still isolated).
    segments = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+|\n+", text.strip())]
    segments = [seg for seg in segments if seg.strip("*-_ ")]
    if not segments:
        return text.strip()

    # Drop a leading preamble segment (short opener, usually ending with ':').
    if preamble.match(segments[0]) and (segments[0].rstrip().endswith(":") or len(segments[0]) < 70):
        segments = segments[1:]

    # Drop trailing meta-commentary segments (process/format self-notes).
    while len(segments) > 1 and meta.search(segments[-1]):
        segments.pop()

    cleaned = " ".join(segments).strip()
    return cleaned or text.strip()


def run_ollama(prompt: str, model: str, temperature: float = 0.2) -> str:
    """Send the prompt to a local Ollama model and return the text reply.

    Low temperature keeps the explanation faithful to the supplied numbers; the
    reply is passed through strip_meta_commentary so small-model chattiness
    (preambles / trailing self-notes) never reaches the officer-facing output.
    """
    import ollama

    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    # New ollama returns an object; older returns a dict — handle both.
    if isinstance(resp, dict):
        content = resp["message"]["content"]
    else:
        content = resp.message.content
    return strip_meta_commentary(content.strip())


def templated_explanation(payload: dict, top_k: int = 5) -> str:
    """Deterministic, no-LLM fallback (used if Ollama is unavailable — Phase 7).

    Still grounded: it mechanically narrates the same top SHAP factors.
    """
    prob = float(payload["meta"]["predicted_prob"])
    rows = payload["rows"][:top_k]
    ups = [r for r in rows if r["shap_contribution"] > 0]
    downs = [r for r in rows if r["shap_contribution"] < 0]
    parts = [f"The model estimates a {prob:.0%} probability of fraud for this transaction."]
    if ups:
        names = ", ".join(f"{r['feature']} ({r['shap_contribution']:+.3f})" for r in ups)
        parts.append(f"The factors raising the risk are: {names}.")
    if downs:
        names = ", ".join(f"{r['feature']} ({r['shap_contribution']:+.3f})" for r in downs)
        parts.append(f"The factors lowering the risk are: {names}.")
    parts.append("This explanation is based only on the listed SHAP factors.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_payload() -> dict:
    """Load the SHAP payload from results/shap_example.json, or regenerate it."""
    if SHAP_JSON.exists():
        return json.loads(SHAP_JSON.read_text(encoding="utf-8"))
    print(f"[ollama] {SHAP_JSON.name} not found — running SHAP (Task 6.1) to create it ...")
    from explain_shap import explain
    res = explain()
    return {
        "meta": res["meta"],
        "raw": res["raw"],
        "rows": [{"rank": i + 1, **r} for i, r in enumerate(res["rows"])],
    }


def explain_nl(model: str = DEFAULT_MODEL, top_k: int = 5, payload: dict | None = None) -> dict:
    """Build the prompt, call Ollama (or fall back), and return prompt + explanation."""
    if payload is None:
        payload = load_payload()
    prompt = build_prompt(payload, top_k=top_k)

    print("[ollama] ===== PROMPT =====")
    print(prompt)
    print("[ollama] ==================\n")

    used_model = None
    try:
        used_model = pick_model(model)
        print(f"[ollama] generating with local model '{used_model}' ...")
        text = run_ollama(prompt, used_model)
        source = f"ollama:{used_model}"
    except Exception as e:
        print(f"[ollama] Ollama unavailable ({type(e).__name__}: {e}). "
              "Using the templated fallback (no LLM).")
        text = templated_explanation(payload, top_k=top_k)
        source = "templated-fallback"

    print("\n[ollama] ===== EXPLANATION =====")
    print(text)
    print("[ollama] =======================")

    _save_markdown(payload, prompt, text, source, top_k)
    return {"prompt": prompt, "explanation": text, "source": source}


def _save_markdown(payload, prompt, text, source, top_k) -> None:
    """Save the prompt + explanation to results/explanation_example.md for the report."""
    raw = payload.get("raw", {})
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = [
        "# NL Explanation — best model (Transformer + GAT)",
        "",
        f"- TransactionID: {raw.get('TransactionID', 'n/a')} "
        f"(true label isFraud={raw.get('isFraud', 'n/a')})",
        f"- Predicted fraud probability: {float(payload['meta']['predicted_prob']):.1%}",
        f"- Explanation source: `{source}`",
        "",
        "## Prompt sent to the local LLM",
        "```",
        prompt,
        "```",
        "",
        "## Generated explanation",
        "",
        text,
        "",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[ollama] wrote {OUT_MD}")


def _select_spotcheck_targets(ctx, n_fraud: int = 3, n_legit: int = 2) -> list[dict]:
    """Pick `n_fraud` confidently-fraud + `n_legit` confidently-legit TEST transactions.

    Frauds = the highest-probability true frauds; legit = the lowest-probability true
    legits. Confident, correctly-classified examples make the spot-check explanations
    clearest (the SHAP signs visibly agree with the prediction direction).
    """
    import numpy as np
    import torch
    from explain_shap import _predict_rows

    model, device = ctx["model"], ctx["device"]
    X_seq, mask, y = ctx["X_seq"], ctx["mask"], ctx["y"]
    graph_emb_all, test_idx = ctx["graph_emb_all"], ctx["test_idx"]

    test_np = test_idx.numpy()
    y_test = y[test_idx].numpy()
    fraud_g = test_np[y_test == 1]
    legit_g = test_np[y_test == 0]

    def probs_for(gids):
        gt = torch.as_tensor(gids, dtype=torch.long)
        return _predict_rows(model, X_seq[gt], mask[gt], gt, graph_emb_all, device)

    fp = probs_for(fraud_g)
    lp = probs_for(legit_g)
    top_fraud = fraud_g[np.argsort(-fp)[:n_fraud]]
    top_legit = legit_g[np.argsort(lp)[:n_legit]]

    targets = [{"global": int(g), "true": 1} for g in top_fraud]
    targets += [{"global": int(g), "true": 0} for g in top_legit]
    return targets


def spot_check(model: str = DEFAULT_MODEL, top_k: int = 5,
               n_fraud: int = 3, n_legit: int = 2, seed: int = config.SEED) -> str:
    """Task 6.3: run the SHAP+Ollama pipeline on 3 fraud + 2 legit transactions and
    save all five grounded explanations to results/explanations_samples.md."""
    from explain_shap import prepare_context, explain_one

    ctx = prepare_context(seed=seed)
    targets = _select_spotcheck_targets(ctx, n_fraud=n_fraud, n_legit=n_legit)
    chosen_model = pick_model(model)
    print(f"[spot] model={chosen_model}; targets (global idx, true): "
          f"{[(t['global'], t['true']) for t in targets]}")

    out = config.RESULTS_DIR / "explanations_samples.md"
    md = [
        "# SHAP + Ollama Spot-Check (Task 6.3)",
        "",
        f"Pipeline: `best_model.pt` (Transformer + GAT) -> SHAP KernelExplainer "
        f"(input-feature level) -> local Ollama `{chosen_model}` NL explanation.",
        "",
        "For each transaction we list the top-5 signed SHAP factors and the LLM "
        "paragraph. Two objective checks per row:",
        "- **Additive check**: `base + sum(all 32 SHAP)` should equal the model's "
        "predicted probability (SHAP is exactly additive here).",
        "- **Sign-consistent**: the net SHAP push direction matches `sign(prob - base)`.",
        "",
        "Frauds are the most-confident true frauds; legits the most-confident true "
        "legits (so the explanations are clear, correctly-classified examples).",
        "",
    ]

    results = []
    for i, t in enumerate(targets, 1):
        kind = "FRAUD" if t["true"] == 1 else "LEGIT"
        print(f"\n[spot] === {i}/{len(targets)} {kind} (global {t['global']}) ===")
        payload = explain_one(t["global"], ctx, save_artifacts=False, verbose=True)
        prompt = build_prompt(payload, top_k=top_k)
        try:
            text = run_ollama(prompt, chosen_model)
            source = f"ollama:{chosen_model}"
        except Exception as e:
            print(f"[spot] Ollama failed ({type(e).__name__}: {e}); templated fallback.")
            text = templated_explanation(payload, top_k=top_k)
            source = "templated-fallback"
        results.append((t, payload, text, source))

        m = payload["meta"]; raw = payload.get("raw", {})
        md += [
            f"## {i}. {kind} — TransactionID {raw.get('TransactionID', 'n/a')} "
            f"(true isFraud={raw.get('isFraud', t['true'])})",
            "",
            f"- Predicted fraud probability: **{m['predicted_prob']:.1%}**",
            f"- Raw: amount={raw.get('TransactionAmt','?')}, hour={raw.get('hour','?')}, "
            f"merchant(ProductCD)={raw.get('ProductCD','?')}, "
            f"device={raw.get('DeviceType','?')}, region(addr1)={raw.get('addr1','?')}",
            f"- base={m['base_value']:.3f} | additive check={m['additive_check']:.3f} | "
            f"net direction=**{m['net_direction']}** | "
            f"sign-consistent=**{m['sign_consistent']}**",
            "",
            "Top-5 SHAP factors (+ raises risk, - lowers):",
            "",
            "| rank | feature | model_value | SHAP |",
            "|---|---|---|---|",
        ]
        for r_i, r in enumerate(payload["rows"][:top_k], 1):
            md.append(f"| {r_i} | {r['feature']} | {r['model_value']:.3f} | "
                      f"{r['shap_contribution']:+.3f} |")
        md += ["", f"NL explanation (`{source}`):", "", f"> {text}", ""]

    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[spot] wrote {out}")

    # Console summary: confirm every row is additive + sign-consistent.
    print("\n[spot] ===== SUMMARY =====")
    all_ok = True
    for t, payload, _txt, _src in results:
        m = payload["meta"]
        ok = m["sign_consistent"] and abs(m["additive_check"] - m["predicted_prob"]) < 1e-4
        all_ok &= ok
        print(f"[spot] global {t['global']:>6} true={t['true']} "
              f"prob={m['predicted_prob']:.3f} net={m['net_direction']:<5} "
              f"sign_ok={m['sign_consistent']} additive_ok="
              f"{abs(m['additive_check'] - m['predicted_prob']) < 1e-4}")
    print(f"[spot] all rows additive + sign-consistent: {all_ok}")
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama NL explanation from SHAP factors (Tasks 6.2/6.3).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Ollama model (default 'llama3.2:3b'; falls back to an installed one).")
    parser.add_argument("--top-k", type=int, default=5,
                        help="number of top SHAP factors to include (default 5).")
    parser.add_argument("--spot-check", action="store_true",
                        help="Task 6.3: explain 3 fraud + 2 legit txns -> explanations_samples.md.")
    args = parser.parse_args()
    if args.spot_check:
        spot_check(model=args.model, top_k=args.top_k)
    else:
        explain_nl(model=args.model, top_k=args.top_k)
