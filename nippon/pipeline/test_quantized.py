"""
test_quantized.py — Test quantized/ONNX models end-to-end, exactly like the SDK.

Production SDK flow per language:
  1. Indic text  → IndicTrans2 (ONNX INT8) → English
  2. English     → IndicTrans2 (ONNX INT8) → back to Indic
  3. Indic text  → MMS-TTS    (ONNX)       → audio .wav

All inference is ONNX Runtime — same path the mobile SDK takes.

Run:
  python test_quantized.py                                     # all languages
  python test_quantized.py --lang hi                           # single language
  python test_quantized.py --lang hi --text "मुझे अकाउंट बनाना है"
"""

import argparse
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

from config import LANGUAGES, RAW_DIR, QUANTIZED_DIR, EXPORT_DIR

TEST_SENTENCES = {
    "hi": "मुझे अकाउंट बनाना है",
    "mr": "मला खाते तयार करायचे आहे",
    "ta": "எனக்கு கணக்கு உருவாக்க வேண்டும்",
    "te": "నాకు ఖాతా సృష్టించాలి",
    "kn": "ನನಗೆ ಖಾತೆ ರಚಿಸಬೇಕು",
    "ml": "എനിക്ക് അക്കൗണ്ട് ഉണ്ടാക്കണം",
    "bn": "আমার অ্যাকাউন্ট তৈরি করতে হবে",
    "gu": "મારે એકાઉન્ટ બનાવવું છે",
    "pa": "ਮੈਨੂੰ ਖਾਤਾ ਬਣਾਉਣਾ ਹੈ",
}


def make_session(ort_path: str) -> InferenceSession:
    opts = SessionOptions()
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    return InferenceSession(ort_path, sess_options=opts)


def greedy_decode(session, input_ids, attention_mask, bos_id, eos_id, max_len=128):
    """Autoregressive greedy decode for IndicTrans2 ONNX — same as on-device."""
    dec = np.array([[bos_id]], dtype=np.int64)
    for _ in range(max_len):
        logits = session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": dec},
        )[0]
        tok = int(logits[0, -1, :].argmax())
        if tok == eos_id:
            break
        dec = np.concatenate([dec, np.array([[tok]], dtype=np.int64)], axis=1)
    return dec[0].tolist()


def translate(session, tokenizer, ip, text, src_lang, tgt_lang, decoder_start_id=2):
    """Full translate: preprocess → ONNX inference → postprocess."""
    batch   = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    enc     = tokenizer(batch, return_tensors="np", padding=True, truncation=True, max_length=256)
    ids     = enc["input_ids"].astype(np.int64)
    mask    = enc["attention_mask"].astype(np.int64)
    eos     = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    out_ids = greedy_decode(session, ids, mask, decoder_start_id, eos)
    raw     = tokenizer.decode(out_ids, skip_special_tokens=True)
    return ip.postprocess_batch([raw], lang=tgt_lang)[0].strip()


def synthesize(tts_session, tts_tokenizer, text, lang_code):
    """TTS via ONNX — returns (audio_array, sample_rate, wav_path)."""
    if not text or len(set(text.strip())) <= 1:
        raise ValueError(f"TTS got invalid text for {lang_code}: '{text[:40]}...'")
    inputs = tts_tokenizer(text, return_tensors="np")
    ids    = inputs["input_ids"].astype(np.int64)
    mask   = inputs.get("attention_mask", np.ones_like(ids)).astype(np.int64)

    waveform = tts_session.run(["waveform"], {"input_ids": ids, "attention_mask": mask})[0]
    audio = waveform[0]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9

    sr       = 16000
    wav_path = Path(f"test_{lang_code}_onnx.wav")
    sf.write(str(wav_path), audio, sr)
    return audio, sr, wav_path


def test_language(lang_code, lang_cfg, text,
                  ie_session, ei_session, ie_tok, ei_tok,
                  ie_dec_start, ei_dec_start,
                  tts_session, tts_tok, ip):
    """Run full SDK flow for one language."""
    name   = lang_cfg["name"]
    flores = lang_cfg["flores"]

    print(f"\n  ┌─ {name} ({lang_code}) ──────────────────────────────────")
    t0 = time.time()

    # Step 1: Indic → English  (IndicTrans2 quantized ONNX)
    english = translate(ie_session, ie_tok, ip, text,
                        src_lang=flores, tgt_lang="eng_Latn",
                        decoder_start_id=ie_dec_start)
    print(f"  │  MT Indic→EN : {text}")
    print(f"  │             → {english}")

    # Step 2: English → Indic  (IndicTrans2 quantized ONNX)
    en_in = english if english and len(english) > 3 else "I need to create an account"
    back = translate(ei_session, ei_tok, ip, en_in,
                     src_lang="eng_Latn", tgt_lang=flores,
                     decoder_start_id=ei_dec_start)
    print(f"  │  MT EN→Indic : {en_in}")
    print(f"  │             → {back}")

    # Step 3: TTS  (MMS-TTS ONNX)
    tts_text = back if back and len(back) > 2 else text
    audio, sr, wav_path = synthesize(tts_session, tts_tok, tts_text, lang_code)
    dur = len(audio) / sr
    print(f"  │  TTS         : {tts_text}")
    print(f"  │             → {wav_path}  ({dur:.1f}s)")

    elapsed = time.time() - t0
    print(f"  └─ done in {elapsed:.1f}s")

    return {
        "input": text,
        "english": english,
        "back": back,
        "wav": str(wav_path),
        "duration_s": round(dur, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Test quantized ONNX models (full SDK flow)")
    parser.add_argument("--lang", default="all",
                        help=f"Language code or 'all'. Available: {list(LANGUAGES.keys())}")
    parser.add_argument("--text", default=None,
                        help="Custom Indic text to translate (used with single --lang)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    print("=" * 62)
    print("  Quantized ONNX — full SDK flow (MT + TTS, all languages)")
    print("  This is exactly how the React Native SDK runs on-device.")
    print("=" * 62)

    from transformers import AutoConfig

    # ── Load MT models (shared across all languages) ──────────────
    ie_onnx = str(Path(QUANTIZED_DIR) / "indic-trans2-indic-en" / "model_quantized.onnx")
    ei_onnx = str(Path(QUANTIZED_DIR) / "indic-trans2-en-indic" / "model_quantized.onnx")

    print("\n[load] IndicTrans2 Indic→EN  (quantized ONNX)")
    ie_session = make_session(ie_onnx)
    print("[load] IndicTrans2 EN→Indic  (quantized ONNX)")
    ei_session = make_session(ei_onnx)

    ie_tok = AutoTokenizer.from_pretrained(f"{RAW_DIR}/indic-trans2-indic-en", trust_remote_code=True)
    ei_tok = AutoTokenizer.from_pretrained(f"{RAW_DIR}/indic-trans2-en-indic", trust_remote_code=True)

    ie_cfg = AutoConfig.from_pretrained(f"{RAW_DIR}/indic-trans2-indic-en", trust_remote_code=True)
    ei_cfg = AutoConfig.from_pretrained(f"{RAW_DIR}/indic-trans2-en-indic", trust_remote_code=True)
    ie_dec_start = ie_cfg.decoder_start_token_id if ie_cfg.decoder_start_token_id is not None else 2
    ei_dec_start = ei_cfg.decoder_start_token_id if ei_cfg.decoder_start_token_id is not None else 2
    print(f"[info] decoder_start_token_id  Indic→EN={ie_dec_start}  EN→Indic={ei_dec_start}")

    ip = IndicProcessor(inference=True)

    # ── Decide which languages ────────────────────────────────────
    if args.lang == "all":
        langs = list(LANGUAGES.keys())
    else:
        if args.lang not in LANGUAGES:
            print(f"Unknown language: {args.lang}. Available: {list(LANGUAGES.keys())}")
            sys.exit(1)
        langs = [args.lang]

    # ── Preload per-language TTS sessions + tokenizers ────────────
    print("[load] MMS-TTS ONNX sessions for each language ...")
    tts_sessions = {}
    tts_tokenizers = {}
    for lc in langs:
        tts_onnx = Path(EXPORT_DIR) / f"mms-tts-{lc}" / "model.onnx"
        if not tts_onnx.exists():
            print(f"  [!] MMS-TTS {lc} — ONNX not found at {tts_onnx}")
            sys.exit(1)
        tts_sessions[lc]   = make_session(str(tts_onnx))
        tts_tokenizers[lc] = AutoTokenizer.from_pretrained(f"{RAW_DIR}/mms-tts-{lc}")
    print(f"[load] {len(langs)} TTS model(s) loaded.\n")

    # ── Run full flow per language ────────────────────────────────
    results = {}
    for lc in langs:
        lang_cfg = LANGUAGES[lc]
        text = args.text if args.text else TEST_SENTENCES.get(lc, lang_cfg["prompt"])

        results[lc] = test_language(
            lc, lang_cfg, text,
            ie_session, ei_session, ie_tok, ei_tok,
            ie_dec_start, ei_dec_start,
            tts_sessions[lc], tts_tokenizers[lc], ip,
        )

    # ── Summary table ─────────────────────────────────────────────
    def trunc(s, n=28):
        return (s[:n] + '…') if len(s) > n else s

    print(f"\n{'='*100}")
    print(f"  Summary — Quantized ONNX (production inference path)")
    print(f"{'='*100}")
    print(f"  {'Lang':<6} {'Input':<28} {'→ English':<28} {'→ Back':<28} {'TTS'}")
    print(f"  {'─'*6} {'─'*28} {'─'*28} {'─'*28} {'─'*10}")
    for lc, r in results.items():
        print(f"  {lc:<6} {trunc(r['input']):<28} {trunc(r['english']):<28} {trunc(r['back']):<28} {r['duration_s']}s")
    print(f"{'='*100}")
    print(f"  Audio files: {', '.join(r['wav'] for r in results.values())}")
    print()


if __name__ == "__main__":
    main()
