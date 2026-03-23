"""
test_e2e.py — End-to-end test using fp32 HuggingFace models.

Shows real results for:
  TTS  → generates audio from prompt
  STT  → transcribes audio back to text
  MT   → Indic to English
  MT   → English to Indic
  TTS  → synthesizes back-translated text

Run:
  python test_e2e.py --lang hi
  python test_e2e.py --lang ta
  python test_e2e.py --lang all
"""

import argparse
import sys
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from config import LANGUAGES, RAW_DIR


def test_language(lang_code: str):
    if lang_code not in LANGUAGES:
        print(f"Unknown language: {lang_code}. Available: {list(LANGUAGES.keys())}")
        return

    lang_cfg = LANGUAGES[lang_code]
    name     = lang_cfg["name"]
    flores   = lang_cfg["flores"]
    prompt   = lang_cfg["prompt"]

    print(f"\n{'='*56}")
    print(f"  End-to-end test — {name} ({lang_code})")
    print(f"{'='*56}")

    # ── Load models ───────────────────────────────────────────
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        VitsModel,
    )
    from IndicTransToolkit import IndicProcessor

    print("\n  Loading models (fp32 HuggingFace — exact same weights as ONNX)...")

    tts_tokenizer = AutoTokenizer.from_pretrained(f"{RAW_DIR}/mms-tts-{lang_code}")
    tts_model     = VitsModel.from_pretrained(f"{RAW_DIR}/mms-tts-{lang_code}").eval()
    tts_sr        = tts_model.config.sampling_rate

    w_processor = WhisperProcessor.from_pretrained(f"{RAW_DIR}/whisper-small")
    w_model     = WhisperForConditionalGeneration.from_pretrained(
        f"{RAW_DIR}/whisper-small"
    ).eval()

    ie_tokenizer = AutoTokenizer.from_pretrained(
        f"{RAW_DIR}/indic-trans2-indic-en", trust_remote_code=True
    )
    ie_model = AutoModelForSeq2SeqLM.from_pretrained(
        f"{RAW_DIR}/indic-trans2-indic-en", trust_remote_code=True
    ).eval()

    ei_tokenizer = AutoTokenizer.from_pretrained(
        f"{RAW_DIR}/indic-trans2-en-indic", trust_remote_code=True
    )
    ei_model = AutoModelForSeq2SeqLM.from_pretrained(
        f"{RAW_DIR}/indic-trans2-en-indic", trust_remote_code=True
    ).eval()

    ip = IndicProcessor(inference=True)
    print("  All models loaded.\n")

    # ── Step 1: TTS ───────────────────────────────────────────
    print(f"  [1] TTS — {name} text → audio")
    print(f"      Input : {prompt}")
    with torch.no_grad():
        tts_inputs = tts_tokenizer(prompt, return_tensors="pt")
        tts_out    = tts_model(**tts_inputs)
        audio      = tts_out.waveform[0].numpy()
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    wav_path = Path(f"test_{lang_code}_prompt.wav")
    sf.write(str(wav_path), audio, tts_sr)
    print(f"      Saved : {wav_path}  ({len(audio)/tts_sr:.1f}s)")

    # ── Step 2: STT ───────────────────────────────────────────
    print(f"\n  [2] STT — audio → {name} text")
    # Resample to 16kHz if needed
    audio_16k = audio
    if tts_sr != 16000:
        import librosa
        audio_16k = librosa.resample(audio, orig_sr=tts_sr, target_sr=16000)

    w_inputs = w_processor(
        audio_16k,
        return_tensors="pt",
        sampling_rate=16000,
    )
    with torch.no_grad():
        w_out = w_model.generate(
            w_inputs.input_features,
            language=lang_cfg["whisper_lang"],
            task="transcribe",
            attention_mask=torch.ones_like(w_inputs.input_features[:, 0, :].long()),
        )
    transcript = w_processor.batch_decode(w_out, skip_special_tokens=True)[0].strip()
    print(f"      Output: {transcript}")

    # Use prompt as fallback if transcript is bad
    mt_input = transcript if transcript and len(transcript) > 3 else prompt

    # ── Step 3: Indic → English ───────────────────────────────
    print(f"\n  [3] MT — {name} → English")
    print(f"      Input : {mt_input}")
    batch_ie = ip.preprocess_batch([mt_input], src_lang=flores, tgt_lang="eng_Latn")
    enc_ie   = ie_tokenizer(
        batch_ie, return_tensors="pt",
        padding=True, truncation=True, max_length=256,
    )
    with torch.no_grad():
        out_ie = ie_model.generate(
            **enc_ie,
            num_beams=1,
            use_cache=False,
            max_new_tokens=256,
        )
    english = ip.postprocess_batch(
        ie_tokenizer.batch_decode(out_ie, skip_special_tokens=True),
        lang="eng_Latn",
    )[0].strip()
    print(f"      Output: {english}")

    # ── Step 4: English → Indic ───────────────────────────────
    en_input = english if english and len(english) > 3 else "Today the weather is very good."
    print(f"\n  [4] MT — English → {name}")
    print(f"      Input : {en_input}")
    batch_ei = ip.preprocess_batch([en_input], src_lang="eng_Latn", tgt_lang=flores)
    enc_ei   = ei_tokenizer(
        batch_ei, return_tensors="pt",
        padding=True, truncation=True, max_length=256,
    )
    with torch.no_grad():
        out_ei = ei_model.generate(
            **enc_ei,
            num_beams=1,
            use_cache=False,
            max_new_tokens=256,
        )
    back = ip.postprocess_batch(
        ei_tokenizer.batch_decode(out_ei, skip_special_tokens=True),
        lang=flores,
    )[0].strip()
    print(f"      Output: {back}")

    # ── Step 5: TTS on back-translated text ───────────────────
    tts_input = back if back and len(back) > 2 else prompt
    print(f"\n  [5] TTS — back-translated {name} → audio")
    print(f"      Input : {tts_input}")
    with torch.no_grad():
        tts_inputs2 = tts_tokenizer(tts_input, return_tensors="pt")
        tts_out2    = tts_model(**tts_inputs2)
        audio2      = tts_out2.waveform[0].numpy()
    if np.max(np.abs(audio2)) > 0:
        audio2 = audio2 / np.max(np.abs(audio2)) * 0.9
    wav_path2 = Path(f"test_{lang_code}_back.wav")
    sf.write(str(wav_path2), audio2, tts_sr)
    print(f"      Saved : {wav_path2}  ({len(audio2)/tts_sr:.1f}s)")

    # ── Summary ───────────────────────────────────────────────
    print(f"""
{'='*56}
  Summary — {name} ({lang_code})
{'='*56}
  [1] TTS input      : {prompt}
      Audio          : {wav_path}
  [2] STT transcript : {transcript}
  [3] Indic→English  : {english}
  [4] English→{name:<10}: {back}
  [5] Back TTS audio : {wav_path2}

  Play audio:
    aplay {wav_path}
    aplay {wav_path2}
{'='*56}
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", default="hi",
        help=f"Language code or 'all'. Available: {list(LANGUAGES.keys())}",
    )
    args = parser.parse_args()

    if args.lang == "all":
        for lc in LANGUAGES:
            test_language(lc)
    else:
        test_language(args.lang)


if __name__ == "__main__":
    main()