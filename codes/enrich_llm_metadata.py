"""
Enrich noisy LLM names using OpenAI API only.

Features:
- only uses OpenAI API
- OpenAI handles web retrieval/search
- configurable batch_size at runtime (default=1)
- configurable limit (process only first N names)
- configurable OpenAI model via CLI
- saves JSON and CSV
- keeps:
    id, original_name, family, training_algorithm, size, domain, language_scope

Install:
    pip install openai pandas tqdm

Set API key:
    export OPENAI_API_KEY=...
or on Windows PowerShell:
    setx OPENAI_API_KEY "your_key"

Examples:
    python enrich_llm_metadata.py --input llm_names.txt
    python enrich_llm_metadata.py --input llm_names.txt --batch_size 10
    python enrich_llm_metadata.py --input llm_names.txt --limit 100
    python enrich_llm_metadata.py --input llm_names.txt --model gpt-5.4
    python enrich_llm_metadata.py --input llm_names.csv --name_col model_name --batch_size 5 --limit 50 --output_prefix out/llm_features

Supported input:
- txt: one model name per line
- json: ["name1", "name2"] or [{"model_name": "..."}]
- csv: specify --name_col if needed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# =========================================================
# 1) Prompt
# =========================================================

SYSTEM_PROMPT = """
You are an expert in open-source LLM ecosystems, Hugging Face models, and model lineage.

Your task:
For each model name, search the web (especially Hugging Face) and infer metadata.

Return ONLY valid JSON.

For each model, output exactly these fields:
- family
- training_algorithm
- size
- domain
- language_scope

Definitions:
1) family:
   The broad model family, e.g. llama, qwen, mistral, gemma, deepseek, falcon, phi, yi, command-r, pythia, olmo, etc.
   If unclear, output "unknown".

2) training_algorithm:
   A short normalized label describing the model's training style / post-training style.
   Use one of:
   - "pretrain"
   - "base"
   - "sft"
   - "instruct"
   - "dpo"
   - "rlhf"
   - "ppo"
   - "distillation"
   - "continued_pretraining"
   - "merge"
   - "unknown"

3) size:
   Normalize into a concise label such as:
   - "0.5B"
   - "1.5B"
   - "3B"
   - "7B"
   - "8B"
   - "13B"
   - "34B"
   - "70B"
   - "671B"
   - "small"
   - "unknown"

   If MoE is clearly indicated, you may use forms like:
   - "8x7B"
   - "16x22B"

4) domain:
   IMPORTANT: domain should only describe task/application specialization,
   NOT language coverage.

   Allowed style examples:
   - "general"
   - "code"
   - "math"
   - "medical"
   - "legal"
   - "finance"
   - "science"
   - "biology"
   - "reasoning"
   - "vision-language"
   - "multimodal"
   - "cybersecurity"
   - "translation"

   Rules for domain:
   - If the model is general-purpose, output "general".
   - Do NOT use language labels such as "multilingual", "portuguese", "chinese", etc. in domain.
   - Language information must go to language_scope instead.

5) language_scope:
   This field describes language coverage or primary language orientation.

   Examples:
   - "english"
   - "multilingual"
   - "monolingual_portuguese"
   - "monolingual_chinese"
   - "monolingual_japanese"
   - "monolingual_korean"
   - "monolingual_arabic"
   - "monolingual_french"
   - "monolingual_german"
   - "unknown"

   Rules for language_scope:
   - If clearly multilingual, output "multilingual".
   - If clearly specialized for one non-English language, use "monolingual_<language_in_english>".
   - If clearly English-centric/general English model, output "english".
   - If unclear, output "unknown".

Important examples:
- A Portuguese chat model for general use:
    domain = "general"
    language_scope = "monolingual_portuguese"

- A multilingual coding model:
    domain = "code"
    language_scope = "multilingual"

- A general English instruct model:
    domain = "general"
    language_scope = "english"

Rules:
- Prefer Hugging Face / model-card style public information when available.
- If the input name is noisy, infer the most likely canonical model behind it.
- Be conservative: if unsure, use "unknown".
- Do not include extra keys.
"""

USER_PROMPT_TEMPLATE = """
Infer metadata for the following LLM names.

Return JSON with EXACTLY this top-level structure:

{{
  "results": [
    {{
      "family": "string",
      "training_algorithm": "string",
      "size": "string",
      "domain": "string",
      "language_scope": "string"
    }}
  ]
}}

Important:
- The number of results MUST equal the number of input names.
- Preserve the same order as the input list.
- Do NOT include original names, ids, explanations, notes, confidence, or any other fields.
- Only include the requested fields.
- domain must be task/application specialization only.
- language information must go into language_scope.

Input names:
{names_json}
"""


# =========================================================
# 2) Helpers
# =========================================================

def normalize_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace("\\", "/")
    s = re.sub(r"\s+", " ", s)
    return s


def load_names(input_path: str, name_col: Optional[str] = None) -> List[str]:
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".txt":
        with open(input_path, "r", encoding="utf-8") as f:
            return [normalize_name(line) for line in f if line.strip()]

    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            if not obj:
                return []

            if isinstance(obj[0], str):
                return [normalize_name(x) for x in obj if str(x).strip()]

            if isinstance(obj[0], dict):
                if name_col is None:
                    for c in ["name", "model_name", "llm_name", "model", "repo_id"]:
                        if c in obj[0]:
                            name_col = c
                            break
                if name_col is None:
                    raise ValueError("JSON is a list of dicts, but no name column found. Please pass --name_col.")
                return [normalize_name(x[name_col]) for x in obj if str(x.get(name_col, "")).strip()]

        raise ValueError("Unsupported JSON format.")

    if ext == ".csv":
        df = pd.read_csv(input_path)
        if len(df.columns) == 0:
            return []
        if name_col is None:
            name_col = df.columns[0]
        return [normalize_name(x) for x in df[name_col].astype(str).tolist() if str(x).strip()]

    raise ValueError(f"Unsupported input file type: {ext}")


def _extract_text_from_response(resp: Any) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    try:
        chunks = []
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []):
                    ctype = getattr(c, "type", None)
                    if ctype in ("output_text", "text"):
                        text = getattr(c, "text", None)
                        if text:
                            chunks.append(text)
        if chunks:
            return "".join(chunks)
    except Exception:
        pass

    return str(resp)


def _try_response_with_tool(
    client: OpenAI,
    model: str,
    prompt: str,
    tool_type: str,
    max_output_tokens: int = 4000,
) -> str:
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        tools=[{"type": tool_type}],
        tool_choice="auto",
        max_output_tokens=max_output_tokens,
    )
    return _extract_text_from_response(response)


def call_openai_batch(
    client: OpenAI,
    names_batch: List[str],
    model: str,
    max_retries: int = 3,
    sleep_sec: float = 2.0,
) -> List[Dict[str, str]]:
    prompt = USER_PROMPT_TEMPLATE.format(
        names_json=json.dumps(names_batch, ensure_ascii=False, indent=2)
    )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            text = None
            tool_errors = []

            for tool_type in ["web_search", "web_search_preview"]:
                try:
                    text = _try_response_with_tool(
                        client=client,
                        model=model,
                        prompt=prompt,
                        tool_type=tool_type,
                    )
                    if text:
                        break
                except Exception as e:
                    tool_errors.append(f"{tool_type}: {e}")

            if not text:
                raise RuntimeError("All web-search tool attempts failed: " + " | ".join(tool_errors))

            data = json.loads(text)

            if "results" not in data or not isinstance(data["results"], list):
                raise ValueError("Model output JSON missing top-level 'results' list.")

            results = data["results"]
            if len(results) != len(names_batch):
                raise ValueError(f"Expected {len(names_batch)} results, got {len(results)}.")

            cleaned = []
            for item in results:
                cleaned.append({
                    "family": str(item.get("family", "unknown")).strip() or "unknown",
                    "training_algorithm": str(item.get("training_algorithm", "unknown")).strip() or "unknown",
                    "size": str(item.get("size", "unknown")).strip() or "unknown",
                    "domain": str(item.get("domain", "general")).strip() or "general",
                    "language_scope": str(item.get("language_scope", "unknown")).strip() or "unknown",
                })
            return cleaned

        except Exception as e:
            last_err = RuntimeError(f"{type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(sleep_sec * attempt)
            else:
                raise last_err


def save_checkpoint(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 3) Main pipeline
# =========================================================

def run_enrichment(
    names: List[str],
    output_prefix: str = "llm_features",
    model: str = "gpt-5.4",
    batch_size: int = 1,
    limit: Optional[int] = None,
    sleep_between_batches: float = 1.0,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    if limit is not None:
        names = names[:limit]

    if len(names) == 0:
        raise ValueError("No model names to process.")

    client = OpenAI()

    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = f"{output_prefix}.checkpoint.json"
    json_path = f"{output_prefix}.json"
    csv_path = f"{output_prefix}.csv"

    results: List[Dict[str, Any]] = load_checkpoint(checkpoint_path)
    start_idx = len(results)

    if start_idx > len(names):
        raise ValueError("Checkpoint has more rows than current input/limit.")

    print(f"Total names to process: {len(names)}")
    print(f"Already completed from checkpoint: {start_idx}")
    print(f"Model: {model}")
    print(f"Batch size: {batch_size}")
    print(f"Output prefix: {output_prefix}")

    total_batches = math.ceil((len(names) - start_idx) / batch_size) if len(names) > start_idx else 0
    batch_counter = 0

    for i in tqdm(range(start_idx, len(names), batch_size), total=total_batches):
        batch = names[i:i + batch_size]

        try:
            batch_results = call_openai_batch(
                client=client,
                names_batch=batch,
                model=model,
            )
        except Exception as e:
            print(f"\nBatch failed at rows {i}..{i + len(batch) - 1}: {e}")
            print("Falling back to per-item calls for this batch...")

            batch_results = []
            for name in batch:
                try:
                    one = call_openai_batch(
                        client=client,
                        names_batch=[name],
                        model=model,
                    )[0]
                except Exception as e_single:
                    print(f"  Failed on '{name}': {e_single}")
                    one = {
                        "family": "unknown",
                        "training_algorithm": "unknown",
                        "size": "unknown",
                        "domain": "general",
                        "language_scope": "unknown",
                    }
                batch_results.append(one)
                time.sleep(0.5)

        for j, item in enumerate(batch_results):
            idx = i + j
            results.append({
                "id": idx,
                "original_name": names[idx],
                "family": item["family"],
                "training_algorithm": item["training_algorithm"],
                "size": item["size"],
                "domain": item["domain"],
                "language_scope": item["language_scope"],
            })

        batch_counter += 1

        if batch_counter % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, results)

        time.sleep(sleep_between_batches)

    save_checkpoint(checkpoint_path, results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(
        results,
        columns=[
            "id",
            "original_name",
            "family",
            "training_algorithm",
            "size",
            "domain",
            "language_scope",
        ]
    )
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")
    print(f"Checkpoint: {checkpoint_path}")

    return df


# =========================================================
# 4) CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Enrich noisy LLM names using OpenAI API only")
    parser.add_argument("--input", required=True, help="Input file: .txt / .json / .csv")
    parser.add_argument("--name_col", default=None, help="Column name for CSV/JSON-dict input")
    parser.add_argument("--output_prefix", default="llm_features", help="Output prefix, without suffix")
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model name, e.g. gpt-5.4 / gpt-4.1 / gpt-4.1-mini")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for each API call. Default: 1")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N names")
    parser.add_argument("--sleep_between_batches", type=float, default=1.0, help="Delay between batches")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Save checkpoint every N batches")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    names = load_names(args.input, name_col=args.name_col)

    run_enrichment(
        names=names,
        output_prefix=args.output_prefix,
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        sleep_between_batches=args.sleep_between_batches,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()