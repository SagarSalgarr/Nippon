"""
test_sdk_from_s3.py — Download models from S3 CDN and run the full SDK inference flow.

Simulates exactly what the React Native SDK does on a phone:
  1. Fetch manifest.json from CDN
  2. Download each model zip (skip if already cached in sdk_cache/)
  3. Extract zips into sdk_cache/
  4. Run full loop per language:
       Indic text → IndicTrans2 → English → IndicTrans2 → Indic → MMS-TTS → .wav
       .wav → Whisper STT → transcript  (closes the full speech round-trip)
  5. Print the summary table

Download strategy (mirrors the APK):
  - On first launch:  whisper + indic_en + indic_from_en  (shared, ~590 MB)
  - On lang select:   mms-tts-{lc}  (per-language, ~105 MB each)

Usage:
  python test_sdk_from_s3.py                    # all 9 languages
  python test_sdk_from_s3.py --lang hi           # single language
  python test_sdk_from_s3.py --no-download       # skip download, use existing sdk_cache/
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import unicodedata

import numpy as np
import soundfile as sf
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

from config import LANGUAGES

# ── S3 CDN ────────────────────────────────────────────────────────────────────
MANIFEST_URL = "https://indicai-cdn.s3.ap-south-1.amazonaws.com/manifest.json"
S3_BASE_URL  = "https://indicai-cdn.s3.ap-south-1.amazonaws.com/models"
CACHE_DIR    = Path("sdk_cache")          # where zips are extracted
RAW_DIR      = Path("models/raw")         # tokenizers for IndicTrans2 (not in zip)

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


# ── Download helpers ───────────────────────────────────────────────────────────

def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def download_file(url: str, dest: Path, expected_sha: str = None):
    """Stream download to dest with progress bar; verify SHA256 if provided."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    total = 0
    start = time.time()
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            sha.update(chunk)
            total += len(chunk)
            mb = total / 1e6
            speed = mb / max(time.time() - start, 0.001)
            print(f"\r    {mb:.1f} MB  ({speed:.1f} MB/s)", end="", flush=True)
    print()
    if expected_sha and sha.hexdigest() != expected_sha:
        dest.unlink()
        raise RuntimeError(f"SHA256 mismatch for {dest.name}")


def is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path):
            return True
    except Exception:
        return False


def extract_zip(zip_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


# ── ONNX helpers ───────────────────────────────────────────────────────────────

def make_session(path: str) -> InferenceSession:
    opts = SessionOptions()
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    return InferenceSession(path, sess_options=opts)


def greedy_decode(session, input_ids, attention_mask, bos_id, eos_id, max_len=128):
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


def translate(session, tokenizer, ip, text, src_lang, tgt_lang, dec_start=2):
    batch   = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    enc     = tokenizer(batch, return_tensors="np", padding=True, truncation=True, max_length=256)
    ids     = enc["input_ids"].astype(np.int64)
    mask    = enc["attention_mask"].astype(np.int64)
    eos     = tokenizer.eos_token_id or 2
    out_ids = greedy_decode(session, ids, mask, dec_start, eos)
    raw     = tokenizer.decode(out_ids, skip_special_tokens=True)
    return ip.postprocess_batch([raw], lang=tgt_lang)[0].strip()


def synthesize(tts_session, tts_tokenizer, text, lang_code):
    inputs   = tts_tokenizer(text, return_tensors="np")
    ids      = inputs["input_ids"].astype(np.int64)
    mask     = inputs.get("attention_mask", np.ones_like(ids)).astype(np.int64)
    waveform = tts_session.run(["waveform"], {"input_ids": ids, "attention_mask": mask})[0]
    audio    = waveform[0]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    sr       = 16000
    wav_path = Path(f"test_{lang_code}_onnx.wav")
    sf.write(str(wav_path), audio, sr)
    return audio, sr, wav_path


def load_intent_engine(onnx_path: Path, vocab_path: Path, bank_path: Path, labels_path: Path):
    """Load MiniLM semantic intent engine. Returns (session, vocab, bank_vecs, bank_labels) or None."""
    if not onnx_path.exists():
        return None
    session     = make_session(str(onnx_path))
    vocab       = {}
    with open(vocab_path) as f:
        for idx, line in enumerate(f):
            vocab[line.rstrip("\n")] = idx
    bank_vecs   = np.load(str(bank_path))                  # (N, 384) float32
    bank_labels = json.loads(labels_path.read_text())      # ["check_balance", ...]
    return session, vocab, bank_vecs, bank_labels


def classify_intent(engine, text: str) -> tuple[str, float]:
    """WordPiece tokenize → MiniLM ONNX encoder → cosine similarity → (intent_label, similarity)."""
    if engine is None:
        return "—", 0.0
    session, vocab, bank_vecs, bank_labels = engine
    MAX_LEN = 64
    CLS, SEP, UNK, PAD = 101, 102, 100, 0

    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    words = norm.split()

    pieces = [CLS]
    for word in words:
        if len(pieces) >= MAX_LEN - 1:
            break
        start, unk_found, subs = 0, False, []
        while start < len(word):
            end = len(word); found = None
            while start < end:
                sub = word[start:end] if start == 0 else f"##{word[start:end]}"
                if sub in vocab: found = vocab[sub]; break
                end -= 1
            if found is None: unk_found = True; break
            subs.append(found); start = end
        if unk_found:
            if len(pieces) < MAX_LEN - 1: pieces.append(UNK)
        else:
            for tok in subs:
                if len(pieces) >= MAX_LEN - 1: break
                pieces.append(tok)
    pieces.append(SEP)

    ids  = np.array([[pieces[i] if i < len(pieces) else PAD for i in range(MAX_LEN)]], dtype=np.int64)
    mask = np.array([[1 if ids[0,i] != PAD else 0 for i in range(MAX_LEN)]], dtype=np.int64)

    # Encoder outputs L2-normalised 384-dim embedding
    emb  = session.run(["sentence_embedding"], {"input_ids": ids, "attention_mask": mask})[0][0]

    # Cosine similarity — all vecs are normalised, so dot = cosine
    sims = bank_vecs @ emb
    idx  = int(np.argmax(sims))
    return bank_labels[idx], float(sims[idx])


def transcribe_onnx(enc_session, dec_session, processor, audio, whisper_lang, max_len=128):
    """Whisper ONNX STT: audio array (16 kHz) → Indic transcript string.

    Encoder:  input_features (1, 80, 3000) → last_hidden_state (1, 1500, 768)
    Decoder:  input_ids + encoder_hidden_states → logits (1, seq, 51865)
    Forced decoder tokens: <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>
    """
    # Feature extraction (mel spectrogram) — pure numpy via WhisperProcessor
    feats = processor(audio, sampling_rate=16000, return_tensors="np").input_features
    feats = feats.astype(np.float32)

    # Encoder
    enc_out = enc_session.run(["last_hidden_state"], {"input_features": feats})[0]

    # Forced prompt tokens
    forced = processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")
    # forced is list of (idx, token_id) — extract token_ids in order
    prompt_ids = [processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
    prompt_ids += [tok for _, tok in forced]

    dec_ids = np.array([prompt_ids], dtype=np.int64)
    eos_id  = processor.tokenizer.eos_token_id  # 50257

    for _ in range(max_len):
        logits = dec_session.run(
            ["logits"],
            {"input_ids": dec_ids, "encoder_hidden_states": enc_out},
        )[0]
        next_tok = int(logits[0, -1, :].argmax())
        if next_tok == eos_id:
            break
        dec_ids = np.concatenate([dec_ids, np.array([[next_tok]], dtype=np.int64)], axis=1)

    # Decode — skip everything up to and including the forced prompt
    generated = dec_ids[0, len(prompt_ids):].tolist()
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Download + extract all needed models ──────────────────────────────────────

def ensure_models(langs: list, skip_download: bool) -> dict:
    """
    Returns paths dict:
      {
        "ie_onnx":  Path,   # Indic→EN quantized ONNX
        "ei_onnx":  Path,   # EN→Indic quantized ONNX
        "ie_raw":   Path,   # Indic→EN tokenizer dir
        "ei_raw":   Path,   # EN→Indic tokenizer dir
        "tts": { lc: Path }  # per-lang TTS dir (has model.onnx + tokenizer)
      }
    """
    if not skip_download:
        print(f"\n[manifest] Fetching {MANIFEST_URL}")
        manifest = fetch_json(MANIFEST_URL)
        print(f"  version={manifest['version']}  languages={list(manifest['languages'].keys())}")

        models_info = manifest["models"]

        def get_zip(filename: str, sha: str = None) -> Path:
            """Return a valid zip at CACHE_DIR/filename.
            Priority: valid cache → dist/ copy → S3 download."""
            cache_zip = CACHE_DIR / filename
            dist_zip  = Path("dist") / filename

            if cache_zip.exists() and is_valid_zip(cache_zip):
                print(f"  [cache] {filename} — OK")
                return cache_zip

            if cache_zip.exists():
                print(f"  [cache] {filename} — corrupt, removing.")
                cache_zip.unlink()

            if dist_zip.exists() and is_valid_zip(dist_zip):
                import shutil
                print(f"  [dist]  {filename} — copying from dist/")
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dist_zip, cache_zip)
                return cache_zip

            url = f"{S3_BASE_URL}/{filename}"
            print(f"  [download] {filename}  ({info.get('size_mb', '?')} MB)")
            download_file(url, cache_zip, sha)
            return cache_zip

        # whisper (downloaded first, like the APK does on first launch)
        info     = models_info["whisper"]
        zip_dest = get_zip(info["file"], info.get("sha256"))
        out_dir  = CACHE_DIR / "whisper-small"
        if not (out_dir / "encoder_model_quantized.onnx").exists():
            print(f"  [extract] {info['file']} → {out_dir}")
            extract_zip(zip_dest, out_dir)

        # shared MT models
        for key, local_name in [("indic_en", "indic-trans2-indic-en"),
                                 ("indic_from_en", "indic-trans2-en-indic")]:
            info     = models_info[key]
            zip_dest = get_zip(info["file"], info.get("sha256"))

            out_dir   = CACHE_DIR / local_name
            onnx_file = out_dir / "model_quantized.onnx"
            if not onnx_file.exists():
                print(f"  [extract] {info['file']} → {out_dir}")
                extract_zip(zip_dest, out_dir)

        # per-language TTS models
        for lc in langs:
            lang_manifest = manifest["languages"].get(lc, {})
            tts = lang_manifest.get("tts")
            if not tts:
                print(f"  [skip] no TTS for {lc} in manifest")
                continue
            info     = tts
            zip_dest = get_zip(tts["file"], tts.get("sha256"))

            out_dir = CACHE_DIR / f"mms-tts-{lc}"
            if not (out_dir / "model.onnx").exists():
                print(f"  [extract] {tts['file']} → {out_dir}")
                extract_zip(zip_dest, out_dir)

        # intent model (optional — present in manifest v3+)
        intent_info = models_info.get("intent")
        if intent_info:
            info     = intent_info
            zip_dest = get_zip(info["file"], info.get("sha256"))
            out_dir  = CACHE_DIR / "intent-minilm"
            if not (out_dir / "intent_encoder.onnx").exists():
                print(f"  [extract] {info['file']} → {out_dir}")
                extract_zip(zip_dest, out_dir)

    # build paths dict
    intent_dir = CACHE_DIR / "intent-minilm"
    paths = {
        "whisper_enc":    CACHE_DIR / "whisper-small" / "encoder_model_quantized.onnx",
        "whisper_dec":    CACHE_DIR / "whisper-small" / "decoder_model_quantized.onnx",
        "whisper_raw":    RAW_DIR   / "whisper-small",
        "ie_onnx":        CACHE_DIR / "indic-trans2-indic-en" / "model_quantized.onnx",
        "ei_onnx":        CACHE_DIR / "indic-trans2-en-indic"  / "model_quantized.onnx",
        "ie_raw":         RAW_DIR   / "indic-trans2-indic-en",
        "ei_raw":         RAW_DIR   / "indic-trans2-en-indic",
        "tts":            {lc: CACHE_DIR / f"mms-tts-{lc}" for lc in langs},
        "intent_onnx":    intent_dir / "intent_encoder.onnx",
        "intent_vocab":   intent_dir / "tokenizer" / "vocab.txt",
        "intent_bank":    intent_dir / "intent_bank.npy",
        "intent_labels":  intent_dir / "intent_bank_labels.json",
    }

    # verify critical files exist
    for label, p in [("Indic→EN ONNX", paths["ie_onnx"]),
                     ("EN→Indic ONNX", paths["ei_onnx"]),
                     ("Indic→EN tokenizer", paths["ie_raw"]),
                     ("EN→Indic tokenizer", paths["ei_raw"])]:
        if not Path(p).exists():
            print(f"\n[ERROR] {label} not found at {p}")
            print("  Run without --no-download to fetch from S3, or check models/raw/")
            sys.exit(1)

    for lc, tts_dir in paths["tts"].items():
        if not (tts_dir / "model.onnx").exists():
            print(f"\n[ERROR] TTS ONNX for {lc} not found at {tts_dir}/model.onnx")
            sys.exit(1)

    return paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download from S3 + run full SDK inference")
    parser.add_argument("--lang", default="all",
                        help=f"Language code or 'all'. Options: {list(LANGUAGES.keys())}")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download, use existing sdk_cache/")
    args = parser.parse_args()

    langs = list(LANGUAGES.keys()) if args.lang == "all" else [args.lang]
    if args.lang != "all" and args.lang not in LANGUAGES:
        print(f"Unknown language: {args.lang}. Available: {list(LANGUAGES.keys())}")
        sys.exit(1)

    print("=" * 68)
    print("  SDK S3 → Inference Test  (download → extract → STT + MT + TTS)")
    print("=" * 68)

    # ── Step 1: Download / verify models ─────────────────────────
    paths = ensure_models(langs, args.no_download)

    # ── Step 2: Load shared sessions + tokenizers ─────────────────
    from transformers import AutoTokenizer, AutoConfig, WhisperProcessor
    from IndicTransToolkit import IndicProcessor

    print("\n[load] Whisper  encoder + decoder  (INT4 ONNX from sdk_cache/)")
    w_enc_session = make_session(str(paths["whisper_enc"]))
    w_dec_session = make_session(str(paths["whisper_dec"]))
    w_processor   = WhisperProcessor.from_pretrained(str(paths["whisper_raw"]))

    print("[load] IndicTrans2  Indic→EN  (INT8 ONNX from sdk_cache/)")
    ie_session = make_session(str(paths["ie_onnx"]))
    print("[load] IndicTrans2  EN→Indic  (INT8 ONNX from sdk_cache/)")
    ei_session = make_session(str(paths["ei_onnx"]))

    ie_tok = AutoTokenizer.from_pretrained(str(paths["ie_raw"]), trust_remote_code=True)
    ei_tok = AutoTokenizer.from_pretrained(str(paths["ei_raw"]), trust_remote_code=True)

    ie_cfg = AutoConfig.from_pretrained(str(paths["ie_raw"]), trust_remote_code=True)
    ei_cfg = AutoConfig.from_pretrained(str(paths["ei_raw"]), trust_remote_code=True)
    ie_dec = ie_cfg.decoder_start_token_id or 2
    ei_dec = ei_cfg.decoder_start_token_id or 2

    ip = IndicProcessor(inference=True)

    # ── Step 3: Load per-language TTS sessions ────────────────────
    # ONNX from sdk_cache/ (downloaded from S3); tokenizer from models/raw/
    # (most zips only bundle model.onnx, not the full tokenizer files)
    print(f"[load] MMS-TTS ONNX for {langs} (from sdk_cache/)")
    tts_sessions    = {}
    tts_tokenizers  = {}
    for lc in langs:
        tts_dir      = paths["tts"][lc]
        tts_tok_dir  = tts_dir if (tts_dir / "tokenizer_config.json").exists() \
                       else RAW_DIR / f"mms-tts-{lc}"
        tts_sessions[lc]   = make_session(str(tts_dir / "model.onnx"))
        tts_tokenizers[lc] = AutoTokenizer.from_pretrained(str(tts_tok_dir))

    # ── Step 3b: Load intent engine ───────────────────────────────
    print("[load] Intent MiniLM  semantic encoder  (from sdk_cache/)")
    intent_engine = load_intent_engine(
        paths["intent_onnx"],
        paths["intent_vocab"],
        paths["intent_bank"],
        paths["intent_labels"],
    )
    if intent_engine is None:
        print("  [skip] intent model not found — intent column will show '—'")

    # ── Step 4: Run full pipeline per language ────────────────────
    print()
    results = {}
    for lc in langs:
        cfg    = LANGUAGES[lc]
        text   = TEST_SENTENCES.get(lc, cfg["prompt"])
        flores = cfg["flores"]

        print(f"  ┌─ {cfg['name']} ({lc}) {'─'*40}")
        t0 = time.time()

        english = translate(ie_session, ie_tok, ip, text,
                            src_lang=flores, tgt_lang="eng_Latn", dec_start=ie_dec)
        print(f"  │  {text}")
        print(f"  │  → {english}")

        en_in = english if english and len(english) > 3 else "I need to create an account"

        intent_label, intent_conf = classify_intent(intent_engine, en_in)
        print(f"  │  Intent → {intent_label}  ({intent_conf*100:.1f}%)")

        back  = translate(ei_session, ei_tok, ip, en_in,
                          src_lang="eng_Latn", tgt_lang=flores, dec_start=ei_dec)
        print(f"  │  → {back}")

        tts_text          = back if back and len(back) > 2 else text
        audio, sr, wav    = synthesize(tts_sessions[lc], tts_tokenizers[lc], tts_text, lc)
        dur               = len(audio) / sr
        print(f"  │  TTS → {wav}  ({dur:.1f}s audio)")

        # Whisper STT: feed TTS output back → should recover the Indic text
        transcript = transcribe_onnx(w_enc_session, w_dec_session, w_processor,
                                     audio, cfg["whisper_lang"])
        elapsed = time.time() - t0
        print(f"  │  STT → {transcript}")
        print(f"  └─ done in {elapsed:.1f}s")

        results[lc] = {
            "input":      text,
            "english":    english,
            "intent":     intent_label,
            "intent_pct": round(intent_conf * 100, 1),
            "back":       back,
            "wav":        str(wav),
            "duration_s": round(dur, 1),
            "transcript": transcript,
        }

    # ── Step 5: Summary table ─────────────────────────────────────
    def trunc(s, n=22):
        return (s[:n] + "…") if len(s) > n else s

    W = 130
    print(f"\n{'='*W}")
    print(f"  {'Lang':<6} {'Input':<22} {'→ English':<22} {'Intent':<20} {'Conf':>6}  {'→ Back':<22} {'TTS':>5}  {'Whisper STT'}")
    print(f"  {'─'*6} {'─'*22} {'─'*22} {'─'*20} {'─'*6}  {'─'*22} {'─'*5}  {'─'*22}")
    for lc, r in results.items():
        print(f"  {lc:<6} {trunc(r['input']):<22} {trunc(r['english']):<22} "
              f"{r['intent']:<20} {r['intent_pct']:>5.1f}%  "
              f"{trunc(r['back']):<22} {r['duration_s']:>4}s  {trunc(r['transcript'])}")
    print(f"{'='*W}")
    wavs = ", ".join(r["wav"] for r in results.values())
    print(f"  Audio files: {wavs}")
    print()


if __name__ == "__main__":
    main()
