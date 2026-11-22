#!/usr/bin/env python3
"""
Stage 06: Alignment Statistics & Visualisation Generator
IIITH Assignment 2A — Forced Alignment using Montreal Forced Aligner (MFA)

Reads the 6 existing Stage 05 TextGrid files and matching WAV audio files
(strictly read-only), computes comprehensive alignment statistics, and
generates 300-DPI overview and zoomed inspection visualisations.
"""

import csv
import json
import os
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import soundfile as sf
from praatio import textgrid
from scipy.signal import spectrogram

BASE_DIR = Path("/Users/swaruppal/Downloads/Ass_2/Assignment 2a")
CORPUS_DIR = BASE_DIR / "alignment_corpus"
TG_DIR = BASE_DIR / "aligned_textgrids"
VIZ_OVERVIEW_DIR = BASE_DIR / "visualizations" / "overview"
VIZ_DETAILED_DIR = BASE_DIR / "visualizations" / "detailed"

RECORDINGS = [
    ("F2B", "F2BJRLP1"),
    ("F2B", "F2BJRLP2"),
    ("F2B", "F2BJRLP3"),
    ("ISLE_SESS0131", "ISLE_SESS0131_BLOCKD02_01_sprt1"),
    ("ISLE_SESS0131", "ISLE_SESS0131_BLOCKD02_02_sprt1"),
    ("ISLE_SESS0131", "ISLE_SESS0131_BLOCKD02_03_sprt1"),
]

DETAILED_REGIONS = [
    {
        "filename": "F2BJRLP1_wbur_zoom.png",
        "speaker": "F2B",
        "recording": "F2BJRLP1",
        "t_start": 20.25,
        "t_end": 21.75,
        "title": "F2BJRLP1 — Zoomed Alignment: \"WBUR's\" (20.56–21.44 s) & Compressed AH0 (20.73–20.74 s)",
        "highlight_words": ["wbur's", "wbur"],
        "highlight_phones_range": [(20.725, 20.745, "AH0 (10 ms)")],
    },
    {
        "filename": "F2BJRLP1_sjcs_zoom.png",
        "speaker": "F2B",
        "recording": "F2BJRLP1",
        "t_start": 4.50,
        "t_end": 5.75,
        "title": "F2BJRLP1 — Zoomed Alignment: Initialism \"S.J.C.'s\" (4.79–5.43 s)",
        "highlight_words": ["s", "j", "c's"],
        "highlight_phones_range": [],
    },
    {
        "filename": "F2BJRLP2_dukakis_zoom.png",
        "speaker": "F2B",
        "recording": "F2BJRLP2",
        "t_start": 2.40,
        "t_end": 3.65,
        "title": "F2BJRLP2 — Zoomed Alignment: Proper Name \"Dukakis\" (2.72–3.36 s)",
        "highlight_words": ["dukakis"],
        "highlight_phones_range": [],
    },
    {
        "filename": "F2BJRLP2_depoliticize_zoom.png",
        "speaker": "F2B",
        "recording": "F2BJRLP2",
        "t_start": 5.15,
        "t_end": 6.80,
        "title": "F2BJRLP2 — Zoomed Alignment: Hyphenated Compound \"de-politicize\" (5.42–6.51 s)",
        "highlight_words": ["de", "politicize"],
        "highlight_phones_range": [],
    },
    {
        "filename": "F2BJRLP2_maffy_zoom.png",
        "speaker": "F2B",
        "recording": "F2BJRLP2",
        "t_start": 23.60,
        "t_end": 24.65,
        "title": "F2BJRLP2 — Zoomed Alignment: Proper Name \"Maffy\" (23.94–24.33 s)",
        "highlight_words": ["maffy"],
        "highlight_phones_range": [],
    },
    {
        "filename": "ISLE_02_no_OW1_zoom.png",
        "speaker": "ISLE_SESS0131",
        "recording": "ISLE_SESS0131_BLOCKD02_02_sprt1",
        "t_start": 1.20,
        "t_end": 2.60,
        "title": "ISLE_SESS0131_BLOCKD02_02_sprt1 — Zoomed Alignment: Final Vowel OW1 in \"NO\" (1.87–2.22 s)",
        "highlight_words": ["no"],
        "highlight_phones_range": [(1.865, 2.225, "OW1 (350 ms)")],
    },
    {
        "filename": "ISLE_03_final_stops_zoom.png",
        "speaker": "ISLE_SESS0131",
        "recording": "ISLE_SESS0131_BLOCKD02_03_sprt1",
        "t_start": 0.45,
        "t_end": 2.40,
        "title": "ISLE_SESS0131_BLOCKD02_03_sprt1 — Zoomed Alignment: Word-Final Stops in \"BAD\", \"NOT\", \"BED\"",
        "highlight_words": ["bad", "not", "bed"],
        "highlight_phones_range": [
            (0.915, 1.175, "D (250 ms)"),
            (1.325, 1.575, "T (240 ms)"),
            (1.795, 2.065, "D (260 ms)"),
        ],
    },
]


def load_tier_intervals(tg, tier_name):
    """Return full list of (start, end, label) including empty silence intervals."""
    tier = tg.getTier(tier_name)
    entries = tier.entries
    full = []
    cur = tg.minTimestamp
    for start, end, label in entries:
        if start > cur + 1e-6:
            full.append((cur, start, ""))
        full.append((start, end, label))
        cur = end
    if tg.maxTimestamp > cur + 1e-6:
        full.append((cur, tg.maxTimestamp, ""))
    return full


def check_structural_anomalies(tg, audio_dur):
    anomalies = []
    if set(tg.tierNames) != {"words", "phones"}:
        anomalies.append(f"Unexpected tiers: {tg.tierNames}")
    if abs(tg.maxTimestamp - audio_dur) > 0.01:
        anomalies.append(f"Duration mismatch: tg={tg.maxTimestamp:.4f}s vs audio={audio_dur:.4f}s")
    for tname in ("words", "phones"):
        tier = tg.getTier(tname)
        prev_end = None
        for s, e, lbl in tier.entries:
            if e <= s:
                anomalies.append(f"{tname} non-positive interval [{s:.4f}, {e:.4f}] '{lbl}'")
            if prev_end is not None and s < prev_end - 1e-6:
                anomalies.append(f"{tname} overlap at {s:.4f} < {prev_end:.4f}")
            if lbl.lower() in ("<unk>", "spn", "unk"):
                anomalies.append(f"{tname} unknown/spn token '{lbl}' at [{s:.4f}, {e:.4f}]")
            prev_end = e
    return anomalies


def compute_spectrogram_db(audio, sr, nperseg=128, noverlap=112):
    """Wideband spectrogram (~8 ms window at 16 kHz) similar to Praat default."""
    freqs, times, Sxx = spectrogram(
        audio, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=512, mode="magnitude"
    )
    Sxx_db = 20.0 * np.log10(np.maximum(Sxx, 1e-10))
    vmax = np.percentile(Sxx_db, 99.5)
    vmin = vmax - 55.0
    return freqs, times, Sxx_db, vmin, vmax


def plot_overview(speaker, rec_name, audio, sr, word_intervals, phone_intervals, out_path):
    dur = len(audio) / sr
    is_long = dur > 10.0

    fig_w = 15.0 if is_long else 12.0
    fig_h = 7.2 if is_long else 6.8
    fig, axes = plt.subplots(
        4, 1, figsize=(fig_w, fig_h), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 2.5, 1.35, 1.15], "hspace": 0.08}
    )

    t = np.arange(len(audio)) / sr
    # Downsample waveform slightly for very long files to keep rendering sharp and fast
    step = max(1, len(audio) // 16000)
    axes[0].plot(t[::step], audio[::step], color="#1f4e79", linewidth=0.55)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Amplitude", fontsize=9)
    axes[0].grid(axis="x", alpha=0.25, linestyle="--")
    axes[0].set_title(
        f"{rec_name} ({speaker}) — Full Utterance Overview ({dur:.2f} s)",
        fontsize=12, fontweight="bold", pad=8
    )

    freqs, times, Sxx_db, vmin, vmax = compute_spectrogram_db(audio, sr)
    axes[1].pcolormesh(times, freqs / 1000.0, Sxx_db, shading="auto", cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[1].set_ylim(0, 8.0)
    axes[1].set_ylabel("Freq (kHz)", fontsize=9)
    for s, e, lbl in word_intervals:
        if lbl != "":
            axes[1].axvline(s, color="#2b6cb0", alpha=0.28, linewidth=0.5)
            axes[1].axvline(e, color="#2b6cb0", alpha=0.28, linewidth=0.5)

    # Word tier
    ax_w = axes[2]
    ax_w.set_ylim(0, 1)
    ax_w.set_yticks([])
    ax_w.set_ylabel("Words", fontsize=9)
    non_empty_words = [iv for iv in word_intervals if iv[2] != ""]
    for idx, (s, e, lbl) in enumerate(word_intervals):
        if lbl == "":
            ax_w.add_patch(patches.Rectangle((s, 0), e - s, 1, facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.4))
        else:
            color = "#dbeafe" if idx % 2 == 0 else "#eff6ff"
            ax_w.add_patch(patches.Rectangle((s, 0), e - s, 1, facecolor=color, edgecolor="#1d4ed8", linewidth=0.6))
            mid = 0.5 * (s + e)
            if is_long:
                # Staggered rotated labels for dense 25-30s broadcast speech
                y_pos = 0.50
                ax_w.text(
                    mid, y_pos, lbl, ha="center", va="center",
                    fontsize=5.2, rotation=90, color="#0f172a", clip_on=True
                )
            else:
                ax_w.text(
                    mid, 0.55, lbl, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color="#0f172a"
                )
                ax_w.text(
                    mid, 0.20, f"{(e - s)*1000:.0f} ms", ha="center", va="center",
                    fontsize=7.5, color="#334155"
                )

    # Phone tier
    ax_p = axes[3]
    ax_p.set_ylim(0, 1)
    ax_p.set_yticks([])
    ax_p.set_ylabel("Phones", fontsize=9)
    short_count = 0
    for idx, (s, e, lbl) in enumerate(phone_intervals):
        dur_ms = (e - s) * 1000.0
        if lbl == "":
            ax_p.add_patch(patches.Rectangle((s, 0), e - s, 1, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.3))
        else:
            is_short = dur_ms <= 30.01
            is_long_ph = dur_ms >= 299.99
            if is_short:
                short_count += 1
                fc, ec = "#fecaca", "#dc2626"
            elif is_long_ph:
                fc, ec = "#fde68a", "#d97706"
            else:
                fc = "#ecfdf5" if idx % 2 == 0 else "#d1fae5"
                ec = "#047857"
            ax_p.add_patch(patches.Rectangle((s, 0), e - s, 1, facecolor=fc, edgecolor=ec, linewidth=0.45))
            if not is_long:
                mid = 0.5 * (s + e)
                y_pos = 0.62 if idx % 2 == 0 else 0.38
                ax_p.text(mid, y_pos, lbl, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#064e3b")

    if is_long:
        n_phones = sum(1 for _, _, l in phone_intervals if l != "")
        ax_p.text(
            0.008, 0.85,
            f"Phone boundaries shown ({n_phones} phones; red = ≤30.01 ms [{short_count}], green = >30.01 ms; ±0.01 ms MFA frame-grid tolerance) — see zoomed plots for ARPAbet labels",
            transform=ax_p.transAxes, fontsize=7.2, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#94a3b8", alpha=0.9)
        )

    axes[3].set_xlim(0, dur)
    axes[3].set_xlabel("Time (seconds)", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.08)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_detailed_zoom(cfg, audio, sr, word_intervals, phone_intervals, out_path):
    t0, t1 = cfg["t_start"], cfg["t_end"]
    i0 = max(0, int(t0 * sr))
    i1 = min(len(audio), int(t1 * sr))
    seg = audio[i0:i1]
    t = np.arange(i0, i1) / sr

    fig, axes = plt.subplots(
        4, 1, figsize=(13.5, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 2.7, 1.15, 1.45], "hspace": 0.08}
    )

    # Waveform
    axes[0].plot(t, seg, color="#1f4e79", linewidth=0.8)
    max_amp = max(0.05, float(np.max(np.abs(seg))) * 1.12)
    axes[0].set_ylim(-max_amp, max_amp)
    axes[0].set_ylabel("Amplitude", fontsize=9.5)
    axes[0].grid(axis="x", alpha=0.25, linestyle="--")
    axes[0].set_title(cfg["title"], fontsize=11.5, fontweight="bold", pad=10)

    # Wideband spectrogram
    freqs, times, Sxx_db, vmin, vmax = compute_spectrogram_db(seg, sr, nperseg=128, noverlap=116)
    times_abs = times + (i0 / sr)
    axes[1].pcolormesh(times_abs, freqs / 1000.0, Sxx_db, shading="auto", cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[1].set_ylim(0, 8.0)
    axes[1].set_ylabel("Freq (kHz)", fontsize=9.5)

    # Highlighted phone spans on spectrogram + waveform
    for hs, he, hlabel in cfg["highlight_phones_range"]:
        for ax in (axes[0], axes[1]):
            ax.axvspan(hs, he, color="#ef4444", alpha=0.16, zorder=2)
        axes[1].text(
            0.5 * (hs + he), 7.45, hlabel, ha="center", va="top",
            fontsize=8.5, fontweight="bold", color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef2f2", edgecolor="#ef4444", alpha=0.92)
        )

    # Word tier
    ax_w = axes[2]
    ax_w.set_ylim(0, 1)
    ax_w.set_yticks([])
    ax_w.set_ylabel("Words", fontsize=9.5)

    hl_words = set(w.lower() for w in cfg["highlight_words"])
    vis_words = [(max(s, t0), min(e, t1), s, e, lbl) for s, e, lbl in word_intervals if e > t0 and s < t1]
    for idx, (vs, ve, orig_s, orig_e, lbl) in enumerate(vis_words):
        if lbl == "":
            ax_w.add_patch(patches.Rectangle((vs, 0), ve - vs, 1, facecolor="#f1f5f9", edgecolor="#94a3b8", linewidth=0.6))
            if (ve - vs) > 0.08:
                ax_w.text(0.5 * (vs + ve), 0.5, "sil", ha="center", va="center", fontsize=8, color="#64748b", style="italic")
        else:
            is_hl = lbl.lower() in hl_words
            fc = "#fef08a" if is_hl else ("#dbeafe" if idx % 2 == 0 else "#eff6ff")
            ec = "#b45309" if is_hl else "#1d4ed8"
            lw = 1.4 if is_hl else 0.8
            ax_w.add_patch(patches.Rectangle((vs, 0), ve - vs, 1, facecolor=fc, edgecolor=ec, linewidth=lw))
            # Project word boundaries onto waveform and spectrogram
            if t0 < orig_s < t1:
                axes[0].axvline(orig_s, color="#1d4ed8", alpha=0.45, linewidth=0.9, linestyle="-")
                axes[1].axvline(orig_s, color="#1d4ed8", alpha=0.55, linewidth=0.9, linestyle="-")
            if t0 < orig_e < t1:
                axes[0].axvline(orig_e, color="#1d4ed8", alpha=0.45, linewidth=0.9, linestyle="-")
                axes[1].axvline(orig_e, color="#1d4ed8", alpha=0.55, linewidth=0.9, linestyle="-")

            mid = 0.5 * (vs + ve)
            if (ve - vs) > 0.045:
                ax_w.text(mid, 0.60, lbl, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")
                ax_w.text(mid, 0.22, f"{(orig_e - orig_s)*1000:.0f} ms", ha="center", va="center", fontsize=7.2, color="#334155")

    # Phone tier
    ax_p = axes[3]
    ax_p.set_ylim(0, 1)
    ax_p.set_yticks([])
    ax_p.set_ylabel("Phones", fontsize=9.5)

    vis_phones = [(max(s, t0), min(e, t1), s, e, lbl) for s, e, lbl in phone_intervals if e > t0 and s < t1]
    non_empty_idx = 0
    for vs, ve, orig_s, orig_e, lbl in vis_phones:
        dur_ms = (orig_e - orig_s) * 1000.0
        if lbl == "":
            ax_p.add_patch(patches.Rectangle((vs, 0), ve - vs, 1, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.5))
        else:
            is_short = dur_ms <= 30.01
            is_long_ph = dur_ms >= 239.99
            if is_short:
                fc, ec = "#fecaca", "#dc2626"
            elif is_long_ph:
                fc, ec = "#fde68a", "#d97706"
            else:
                fc = "#ecfdf5" if non_empty_idx % 2 == 0 else "#d1fae5"
                ec = "#047857"
            ax_p.add_patch(patches.Rectangle((vs, 0), ve - vs, 1, facecolor=fc, edgecolor=ec, linewidth=0.75))

            if t0 < orig_s < t1:
                axes[1].axvline(orig_s, color="#059669", alpha=0.35, linewidth=0.6, linestyle=":")

            mid = 0.5 * (vs + ve)
            # 3-row staggering for narrow phones so every label is readable
            row_y = [0.76, 0.48, 0.20][non_empty_idx % 3]
            fs = 7.4 if (ve - vs) < 0.045 else 8.4
            txt = f"{lbl}\n({dur_ms:.0f})" if (ve - vs) >= 0.065 else f"{lbl}"
            ax_p.text(
                mid, row_y, txt, ha="center", va="center",
                fontsize=fs, fontweight="bold",
                color="#991b1b" if is_short else "#064e3b"
            )
            non_empty_idx += 1

    axes[3].set_xlim(t0, t1)
    axes[3].set_xlabel("Time (seconds)", fontsize=10)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.93, bottom=0.08)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summarize_group(name, rec_rows, all_word_durs, all_phone_durs):
    return {
        "group": name,
        "num_recordings": len(rec_rows),
        "total_audio_duration_s": round(sum(r["audio_duration_s"] for r in rec_rows), 4),
        "word_intervals_total": sum(r["word_intervals_total"] for r in rec_rows),
        "word_intervals_non_empty": sum(r["word_intervals_non_empty"] for r in rec_rows),
        "word_silence_intervals": sum(r["word_silence_intervals"] for r in rec_rows),
        "phone_intervals_total": sum(r["phone_intervals_total"] for r in rec_rows),
        "phone_intervals_non_empty": sum(r["phone_intervals_non_empty"] for r in rec_rows),
        "phone_silence_intervals": sum(r["phone_silence_intervals"] for r in rec_rows),
        "mean_word_duration_ms": round(mean(all_word_durs), 2),
        "median_word_duration_ms": round(median(all_word_durs), 2),
        "mean_phone_duration_ms": round(mean(all_phone_durs), 2),
        "median_phone_duration_ms": round(median(all_phone_durs), 2),
        "min_phone_duration_ms": round(min(all_phone_durs), 2),
        "max_phone_duration_ms": round(max(all_phone_durs), 2),
        "phones_shorter_than_nominal_30ms": sum(r["phones_shorter_than_nominal_30ms"] for r in rec_rows),
        "phones_nominally_30ms": sum(r["phones_nominally_30ms"] for r in rec_rows),
        "phones_nominally_30ms_or_shorter": sum(r["phones_nominally_30ms_or_shorter"] for r in rec_rows),
        "phones_at_least_nominally_300ms": sum(r["phones_at_least_nominally_300ms"] for r in rec_rows),
        "speech_rate_words_per_min": round(
            sum(r["word_intervals_non_empty"] for r in rec_rows)
            / (sum(r["audio_duration_s"] for r in rec_rows) / 60.0), 2
        ),
        "structural_anomalies_count": sum(r["structural_anomalies_count"] for r in rec_rows),
    }


def main():
    VIZ_OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DETAILED_DIR.mkdir(parents=True, exist_ok=True)

    loaded_data = {}
    rec_stats = []
    group_word_durs = {"All_6": [], "F2B": [], "ISLE_SESS0131": []}
    group_phone_durs = {"All_6": [], "F2B": [], "ISLE_SESS0131": []}

    for speaker, rec_name in RECORDINGS:
        wav_path = CORPUS_DIR / speaker / f"{rec_name}.wav"
        tg_path = TG_DIR / speaker / f"{rec_name}.TextGrid"

        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio_dur = len(audio) / sr

        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
        word_intervals = load_tier_intervals(tg, "words")
        phone_intervals = load_tier_intervals(tg, "phones")
        anomalies = check_structural_anomalies(tg, audio_dur)

        loaded_data[(speaker, rec_name)] = (audio, sr, word_intervals, phone_intervals)

        ne_words = [(s, e, l) for s, e, l in word_intervals if l != ""]
        ne_phones = [(s, e, l) for s, e, l in phone_intervals if l != ""]
        sil_words = [iv for iv in word_intervals if iv[2] == ""]
        sil_phones = [iv for iv in phone_intervals if iv[2] == ""]

        w_durs = [(e - s) * 1000.0 for s, e, _ in ne_words]
        p_durs = [(e - s) * 1000.0 for s, e, _ in ne_phones]

        group_word_durs["All_6"].extend(w_durs)
        group_word_durs[speaker].extend(w_durs)
        group_phone_durs["All_6"].extend(p_durs)
        group_phone_durs[speaker].extend(p_durs)

        min_p = min(ne_phones, key=lambda x: x[1] - x[0])
        max_p = max(ne_phones, key=lambda x: x[1] - x[0])
        # Tolerance of +/-0.01 ms (+/-10 us) accounts for MFA's +/-1-2 us float32 TextGrid serialization jitter
        # on its 10 ms frame grid, while excluding empty/silence intervals consistently.
        tol_ms = 0.01
        lt_30_phones = [(s, e, l, round((e - s) * 1000.0, 3)) for s, e, l in ne_phones if (e - s) * 1000.0 < (30.0 - tol_ms)]
        eq_30_phones = [(s, e, l, round((e - s) * 1000.0, 3)) for s, e, l in ne_phones if abs((e - s) * 1000.0 - 30.0) <= tol_ms]
        short_phones = [(s, e, l, round((e - s) * 1000.0, 3)) for s, e, l in ne_phones if (e - s) * 1000.0 <= (30.0 + tol_ms)]
        long_phones = [(s, e, l, round((e - s) * 1000.0, 3)) for s, e, l in ne_phones if (e - s) * 1000.0 >= (300.0 - tol_ms)]

        row = {
            "recording": rec_name,
            "speaker_group": speaker,
            "audio_duration_s": round(audio_dur, 4),
            "textgrid_duration_s": round(tg.maxTimestamp, 4),
            "duration_diff_s": round(abs(audio_dur - tg.maxTimestamp), 6),
            "word_intervals_total": len(word_intervals),
            "word_intervals_non_empty": len(ne_words),
            "word_silence_intervals": len(sil_words),
            "phone_intervals_total": len(phone_intervals),
            "phone_intervals_non_empty": len(ne_phones),
            "phone_silence_intervals": len(sil_phones),
            "mean_word_duration_ms": round(mean(w_durs), 2),
            "median_word_duration_ms": round(median(w_durs), 2),
            "min_word_duration_ms": round(min(w_durs), 2),
            "max_word_duration_ms": round(max(w_durs), 2),
            "mean_phone_duration_ms": round(mean(p_durs), 2),
            "median_phone_duration_ms": round(median(p_durs), 2),
            "min_phone_duration_ms": round((min_p[1] - min_p[0]) * 1000.0, 2),
            "min_phone_label": f"{min_p[2]} [{min_p[0]:.3f}-{min_p[1]:.3f}s]",
            "max_phone_duration_ms": round((max_p[1] - max_p[0]) * 1000.0, 2),
            "max_phone_label": f"{max_p[2]} [{max_p[0]:.3f}-{max_p[1]:.3f}s]",
            "phones_shorter_than_nominal_30ms": len(lt_30_phones),
            "phones_nominally_30ms": len(eq_30_phones),
            "phones_nominally_30ms_or_shorter": len(short_phones),
            "phones_at_least_nominally_300ms": len(long_phones),
            "speech_rate_words_per_min": round(len(ne_words) / (audio_dur / 60.0), 2),
            "structural_anomalies_count": len(anomalies),
            "structural_anomalies": "; ".join(anomalies) if anomalies else "None",
        }
        rec_stats.append(row)

        # Generate overview plot
        out_overview = VIZ_OVERVIEW_DIR / f"{rec_name}_overview.png"
        plot_overview(speaker, rec_name, audio, sr, word_intervals, phone_intervals, out_overview)
        print(f"Generated overview: {out_overview.name}")

    # Generate detailed zoomed plots + log exact intervals in each zoomed window
    region_details = {}
    for cfg in DETAILED_REGIONS:
        key = (cfg["speaker"], cfg["recording"])
        audio, sr, word_intervals, phone_intervals = loaded_data[key]
        out_det = VIZ_DETAILED_DIR / cfg["filename"]
        plot_detailed_zoom(cfg, audio, sr, word_intervals, phone_intervals, out_det)
        print(f"Generated detailed: {out_det.name}")

        w_sub = [
            {"word": l, "start_s": round(s, 4), "end_s": round(e, 4), "dur_ms": round((e - s) * 1000, 1)}
            for s, e, l in word_intervals if e > cfg["t_start"] and s < cfg["t_end"]
        ]
        p_sub = [
            {"phone": l, "start_s": round(s, 4), "end_s": round(e, 4), "dur_ms": round((e - s) * 1000, 1)}
            for s, e, l in phone_intervals if e > cfg["t_start"] and s < cfg["t_end"]
        ]
        region_details[cfg["filename"]] = {"words": w_sub, "phones": p_sub}

    # Group aggregates
    f2b_rows = [r for r in rec_stats if r["speaker_group"] == "F2B"]
    isle_rows = [r for r in rec_stats if r["speaker_group"] == "ISLE_SESS0131"]
    group_stats = {
        "All_6": summarize_group("All_6", rec_stats, group_word_durs["All_6"], group_phone_durs["All_6"]),
        "F2B": summarize_group("F2B", f2b_rows, group_word_durs["F2B"], group_phone_durs["F2B"]),
        "ISLE_SESS0131": summarize_group("ISLE_SESS0131", isle_rows, group_word_durs["ISLE_SESS0131"], group_phone_durs["ISLE_SESS0131"]),
    }

    # Save JSON
    stats_json = {
        "counting_definition": {
            "intervals_counted": "Non-empty phone intervals only (empty/silence intervals excluded from duration metrics and threshold counts; silence interval counts reported separately).",
            "tolerance_ms": 0.01,
            "rationale": "Reconciled counts using the MFA 10 ms frame-grid tolerance (+/-0.01 ms = +/-10 us) to accommodate float32 TextGrid timestamp serialization jitter (e.g., 29.998-30.001 ms). The 75 nominally 30 ms intervals include 12 intervals serialized as 30.001 ms; the 2 shorter-than-nominal-30-ms intervals are the two 10 ms AH0 intervals in F2BJRLP1.",
            "classifications": {
                "phones_shorter_than_nominal_30ms": "Shorter than nominal 30 ms (duration < 29.99 ms)",
                "phones_nominally_30ms": "Nominally 30 ms (duration within 29.99-30.01 ms, inclusive)",
                "phones_nominally_30ms_or_shorter": "Nominally 30 ms or shorter (duration <= 30.01 ms)",
                "phones_at_least_nominally_300ms": "At least nominally 300 ms (duration >= 299.99 ms)",
            },
        },
        "per_recording": rec_stats,
        "group_aggregates": group_stats,
        "zoomed_region_intervals": region_details,
    }
    json_path = BASE_DIR / "alignment_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)
    print(f"Saved: {json_path}")

    # Save CSV
    csv_path = BASE_DIR / "alignment_statistics.csv"
    fieldnames = list(rec_stats[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rec_stats)
    print(f"Saved: {csv_path}")

    print("\n=== SUMMARY TABLE ===")
    for r in rec_stats:
        print(
            f"{r['recording']:32s} | dur={r['audio_duration_s']:6.2f}s | "
            f"W={r['word_intervals_non_empty']:2d}/{r['word_intervals_total']:2d} (mean={r['mean_word_duration_ms']:5.1f}ms) | "
            f"P={r['phone_intervals_non_empty']:3d}/{r['phone_intervals_total']:3d} (mean={r['mean_phone_duration_ms']:5.1f}ms, "
            f"min={r['min_phone_duration_ms']:4.1f}ms, max={r['max_phone_duration_ms']:5.1f}ms) | "
            f"<=30.01ms={r['phones_nominally_30ms_or_shorter']:2d} | >=299.99ms={r['phones_at_least_nominally_300ms']} | anom={r['structural_anomalies_count']}"
        )
    print("\n=== GROUP AGGREGATES ===")
    for gname, g in group_stats.items():
        print(
            f"{gname:15s} | dur={g['total_audio_duration_s']:6.2f}s | "
            f"W={g['word_intervals_non_empty']:3d} (mean={g['mean_word_duration_ms']:5.1f}ms, med={g['median_word_duration_ms']:5.1f}ms, {g['speech_rate_words_per_min']:5.1f} wpm) | "
            f"P={g['phone_intervals_non_empty']:4d} (mean={g['mean_phone_duration_ms']:5.1f}ms, med={g['median_phone_duration_ms']:5.1f}ms) | "
            f"<=30.01ms={g['phones_nominally_30ms_or_shorter']:2d} | >=299.99ms={g['phones_at_least_nominally_300ms']}"
        )


if __name__ == "__main__":
    main()
