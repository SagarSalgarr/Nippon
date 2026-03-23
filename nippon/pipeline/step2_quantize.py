"""
step2_quantize.py — Export models to ONNX then quantize.

Quantization strategy:
  Whisper Small : INT4  via optimum main_export + quantize_dynamic
  IndicTrans2   : INT8  via torch.onnx.export (custom arch, optimum unsupported)
  MMS-TTS       : none  via torch.onnx.export (VITS stochastic output breaks optimum validation)

Run:  python step2_quantize.py
"""

import sys
import shutil
import hashlib
import json
import torch
from pathlib import Path

from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType

from config import (
    LANGUAGES,
    MODEL_VERSION,
    RAW_DIR,
    EXPORT_DIR,
    QUANTIZED_DIR,
    DIST_DIR,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    print(f"  {msg}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def quantize_file(src: Path, dst: Path, use_int4: bool = False):
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QUInt4 if use_int4 else QuantType.QInt8,
        extra_options={"DefaultTensorType": 1, "MatMulConstBOnly": True},
    )


def make_zip(source_dir: Path, dist_filename: str) -> dict:
    Path(DIST_DIR).mkdir(parents=True, exist_ok=True)
    zip_base   = Path(DIST_DIR) / dist_filename.replace(".zip", "")
    final_path = Path(str(zip_base) + ".zip")
    if not final_path.exists():
        shutil.make_archive(str(zip_base), "zip", str(source_dir))
    sha     = sha256_file(final_path)
    size_mb = round(final_path.stat().st_size / 1_000_000, 1)
    final_path.with_suffix(final_path.suffix + ".sha256").write_text(sha)
    log(f"[dist]   {final_path.name}  sha={sha[:12]}...  size={size_mb}MB")
    return {"file": final_path.name, "sha256": sha, "size_mb": size_mb, "version": MODEL_VERSION}


def quantize_onnx_folder(export_path: Path, quant_path: Path, model_name: str, use_int4: bool = False):
    onnx_files = [
        f for f in export_path.glob("*.onnx")
        if "quantized" not in f.name and "optimized" not in f.name
    ]
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx files in {export_path}")
    label = "INT4" if use_int4 else "INT8"
    log(f"[quant]  {model_name} — {len(onnx_files)} file(s) -> {label}: {[f.name for f in onnx_files]}")
    quant_path.mkdir(parents=True, exist_ok=True)
    for f in onnx_files:
        out = quant_path / f.name.replace(".onnx", "_quantized.onnx")
        log(f"[quant]  {f.name} -> {label} ...")
        quantize_file(f, out, use_int4=use_int4)
        log(f"[quant]  {out.name} — {round(out.stat().st_size/1e6,1)}MB")


# ── Whisper (INT4, optimum export) ────────────────────────────────────────────

def quantize_whisper() -> dict:
    print("\n── Whisper Small ─────────────────────────────────────────────")
    model_name  = "whisper-small"
    export_path = Path(EXPORT_DIR) / model_name
    quant_path  = Path(QUANTIZED_DIR) / model_name
    dist_file   = f"whisper-small-int4-v{MODEL_VERSION}.zip"

    if export_path.exists() and any(export_path.glob("*.onnx")):
        log(f"[export] {model_name} — already exported, skipping")
    else:
        log(f"[export] {model_name} -> ONNX")
        export_path.mkdir(parents=True, exist_ok=True)
        main_export(
            model_name_or_path=f"{RAW_DIR}/{model_name}",
            output=export_path,
            task="automatic-speech-recognition",
            opset=18,
            optimize="O2",
            trust_remote_code=True,
        )
        log(f"[export] {model_name} — done")

    if quant_path.exists() and any(quant_path.glob("*quantized*.onnx")):
        log(f"[quant]  {model_name} — already quantized, skipping")
    else:
        quantize_onnx_folder(export_path, quant_path, model_name, use_int4=True)

    return make_zip(quant_path, dist_file)


# ── IndicTrans2 (INT8, torch.onnx.export) ─────────────────────────────────────

def export_indictrans(raw_dir: str, export_path: Path, model_name: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from IndicTransToolkit import IndicProcessor

    log(f"[export] loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(raw_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(raw_dir, trust_remote_code=True).float()
    model.eval()

    ip = IndicProcessor(inference=True)
    if "indic-en" in model_name:
        sentences = ["आज मौसम बहुत अच्छा है।"]
        batch = ip.preprocess_batch(sentences, src_lang="hin_Deva", tgt_lang="eng_Latn")
    else:
        sentences = ["Today the weather is very good."]
        batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")

    inputs = tokenizer(batch, return_tensors="pt", padding="longest", truncation=True, max_length=32)
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    bos_token_id   = model.config.decoder_start_token_id or model.config.bos_token_id or 2

    # Trace with tgt_seq > 1 so internal reshapes capture dynamic shapes.
    # Using length 4 forces ONNX to keep tgt_seq dimension symbolic.
    decoder_input_ids = torch.full((1, 4), bos_token_id, dtype=torch.long)

    export_path.mkdir(parents=True, exist_ok=True)
    onnx_path = export_path / "model.onnx"
    log(f"[export] tracing {model_name} -> ONNX (decoder dummy len=4 for dynamic shapes) ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask, decoder_input_ids),
            str(onnx_path),
            opset_version=17,
            dynamo=False,
            input_names=["input_ids", "attention_mask", "decoder_input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":         {0: "batch", 1: "src_seq"},
                "attention_mask":    {0: "batch", 1: "src_seq"},
                "decoder_input_ids": {0: "batch", 1: "tgt_seq"},
                "logits":            {0: "batch", 1: "tgt_seq"},
            },
        )
    log(f"[export] {model_name} — {round(onnx_path.stat().st_size/1e6,1)}MB")


def quantize_indic(raw_dir: str, model_name: str, dist_file: str) -> dict:
    export_path = Path(EXPORT_DIR) / model_name
    quant_path  = Path(QUANTIZED_DIR) / model_name

    if (export_path / "model.onnx").exists():
        size = round((export_path / "model.onnx").stat().st_size / 1e6, 1)
        if size < 10:
            log(f"[export] {model_name} — previous export too small ({size}MB), re-exporting")
            shutil.rmtree(export_path)
            export_indictrans(raw_dir, export_path, model_name)
        else:
            log(f"[export] {model_name} — already exported ({size}MB), skipping")
    else:
        export_indictrans(raw_dir, export_path, model_name)

    if (quant_path / "model_quantized.onnx").exists():
        log(f"[quant]  {model_name} — already quantized, skipping")
    else:
        quant_path.mkdir(parents=True, exist_ok=True)
        log(f"[quant]  {model_name} -> INT8 ...")
        quantize_file(export_path / "model.onnx", quant_path / "model_quantized.onnx", use_int4=False)
        size = round((quant_path / "model_quantized.onnx").stat().st_size / 1e6, 1)
        log(f"[quant]  {model_name} — {size}MB")

    return make_zip(quant_path, dist_file)


def quantize_indic_en() -> dict:
    print("\n── IndicTrans2 Indic->EN ──────────────────────────────────────")
    return quantize_indic(
        raw_dir=f"{RAW_DIR}/indic-trans2-indic-en",
        model_name="indic-trans2-indic-en",
        dist_file=f"indic-trans2-indic-en-int8-v{MODEL_VERSION}.zip",
    )


def quantize_indic_from_en() -> dict:
    print("\n── IndicTrans2 EN->Indic ──────────────────────────────────────")
    return quantize_indic(
        raw_dir=f"{RAW_DIR}/indic-trans2-en-indic",
        model_name="indic-trans2-en-indic",
        dist_file=f"indic-trans2-en-indic-int8-v{MODEL_VERSION}.zip",
    )


# ── MMS-TTS (torch.onnx.export, no quantization) ─────────────────────────────

def export_mms_tts(lang_code: str):
    raw_dir = Path(f"{RAW_DIR}/mms-tts-{lang_code}")
    if not raw_dir.exists():
        log(f"[skip]   MMS-TTS {lang_code} — not downloaded")
        return None

    print(f"\n── MMS-TTS {lang_code} ────────────────────────────────────────────")
    model_name  = f"mms-tts-{lang_code}"
    export_path = Path(EXPORT_DIR) / model_name
    dist_file   = f"mms-tts-{lang_code}-v{MODEL_VERSION}.zip"

    if export_path.exists() and any(export_path.glob("*.onnx")):
        log(f"[export] {model_name} — already exported, skipping")
    else:
        from transformers import VitsModel, AutoTokenizer

        log(f"[export] loading {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(str(raw_dir))
        model     = VitsModel.from_pretrained(str(raw_dir)).float()
        model.eval()

        # Dummy input: short text in the target language script
        lang_cfg  = LANGUAGES.get(lang_code, {})
        # Use the stt_prompt text as dummy TTS input (it's in the right script)
        dummy_text = lang_cfg.get("prompt", "hello")
        inputs = tokenizer(dummy_text, return_tensors="pt")
        input_ids      = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        export_path.mkdir(parents=True, exist_ok=True)
        onnx_path = export_path / "model.onnx"

        log(f"[export] tracing {model_name} -> ONNX (TorchScript path) ...")
        with torch.no_grad():
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                str(onnx_path),
                opset_version=16,   # VITS needs <=16; also forces TorchScript path on PyTorch 2.x
                dynamo=False,       # CRITICAL: VITS has data-dependent control flow incompatible with dynamo
                input_names=["input_ids", "attention_mask"],
                output_names=["waveform"],
                dynamic_axes={
                    "input_ids":      {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "waveform":       {0: "batch", 1: "time"},
                },
            )
        log(f"[export] {model_name} — {round(onnx_path.stat().st_size/1e6,1)}MB")

    return make_zip(export_path, dist_file)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Step 2 — Export + Quantize")
    print("=" * 60)

    manifest_models = {}

    try:
        manifest_models["whisper"] = quantize_whisper()
    except Exception as e:
        print(f"\n[FATAL] Whisper failed: {e}")
        sys.exit(1)

    try:
        manifest_models["indic_en"] = quantize_indic_en()
    except Exception as e:
        print(f"\n[FATAL] IndicTrans2 Indic->EN failed: {e}")
        sys.exit(1)

    try:
        manifest_models["indic_from_en"] = quantize_indic_from_en()
    except Exception as e:
        print(f"\n[FATAL] IndicTrans2 EN->Indic failed: {e}")
        sys.exit(1)

    manifest_models["tts"] = {}
    for lang_code in LANGUAGES:
        result = export_mms_tts(lang_code)
        if result:
            manifest_models["tts"][lang_code] = result

    Path(DIST_DIR).mkdir(parents=True, exist_ok=True)
    draft_path = Path(DIST_DIR) / "manifest_draft.json"
    draft_path.write_text(json.dumps(manifest_models, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("  Quantization complete.")
    print(f"  Output : {DIST_DIR}/")
    print(f"  Draft manifest : {draft_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()