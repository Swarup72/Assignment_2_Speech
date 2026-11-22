#!/usr/bin/env python3
"""
Generate Assignment_2A_Forced_Alignment_Report.pdf using Google Chrome headless.
Embeds 300-DPI PNG visualisations and exact Stage 06 alignment statistics.
"""

import subprocess
import tempfile
from pathlib import Path

BASE_DIR = Path("/Users/swaruppal/Downloads/Ass_2/Assignment 2a")
PDF_OUT = BASE_DIR / "Assignment_2A_Forced_Alignment_Report.pdf"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

FIG_F2B_WBUR = (BASE_DIR / "visualizations/detailed/F2BJRLP1_wbur_zoom.png").as_uri()
FIG_F2B_SJCS = (BASE_DIR / "visualizations/detailed/F2BJRLP1_sjcs_zoom.png").as_uri()
FIG_ISLE_02 = (BASE_DIR / "visualizations/overview/ISLE_SESS0131_BLOCKD02_02_sprt1_overview.png").as_uri()
FIG_ISLE_03 = (BASE_DIR / "visualizations/detailed/ISLE_03_final_stops_zoom.png").as_uri()

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IIITH Assignment 2A — Forced Alignment Report</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 8.5mm 12.5mm 8.5mm 12.5mm;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
    font-size: 8.3pt;
    line-height: 1.32;
    color: #1e293b;
    margin: 0;
    padding: 0;
  }}
  .header {{
    border-bottom: 2px solid #1e3a8a;
    padding-bottom: 6px;
    margin-bottom: 8px;
  }}
  .header h1 {{
    font-size: 14.5pt;
    color: #0f172a;
    margin: 0 0 3px 0;
    letter-spacing: -0.2px;
  }}
  .header .subtitle {{
    font-size: 9.2pt;
    color: #334155;
    margin: 0 0 5px 0;
  }}
  .meta-row {{
    display: flex;
    justify-content: space-between;
    font-size: 8.1pt;
    color: #475569;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 4px 8px;
  }}
  h2 {{
    font-size: 10.6pt;
    color: #1e3a8a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 2px;
    margin: 9px 0 4px 0;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 9.2pt;
    color: #0f172a;
    margin: 7px 0 3px 0;
    page-break-after: avoid;
  }}
  p {{
    margin: 0 0 5px 0;
    text-align: justify;
  }}
  ul {{
    margin: 3px 0 6px 0;
    padding-left: 16px;
  }}
  li {{
    margin-bottom: 2.5px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 4px 0 7px 0;
    font-size: 7.4pt;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 2px 3.5px;
    text-align: left;
  }}
  th {{
    background: #eff6ff;
    color: #1e3a8a;
    font-weight: 600;
  }}
  tr:nth-child(even) td {{
    background: #f8fafc;
  }}
  .num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  code {{
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 8.0pt;
    background: #f1f5f9;
    padding: 0.5px 3px;
    border-radius: 2px;
    color: #0f172a;
  }}
  .callout {{
    background: #fffbeb;
    border-left: 3.5px solid #d97706;
    padding: 4.5px 8px;
    margin: 5px 0 0 0;
    font-size: 8.0pt;
    border-radius: 0 4px 4px 0;
  }}
  .callout-blue {{
    background: #eff6ff;
    border-left: 3.5px solid #2563eb;
    padding: 6px 9px;
    margin: 7px 0 9px 0;
    font-size: 8.6pt;
    border-radius: 0 4px 4px 0;
  }}
  .figure-box {{
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 6px;
    margin: 7px 0 10px 0;
    background: #ffffff;
    page-break-inside: avoid;
  }}
  .figure-box img {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .caption {{
    font-size: 8.1pt;
    color: #334155;
    margin-top: 5px;
    line-height: 1.35;
  }}
  .page-break {{
    page-break-before: always;
  }}
  .two-col {{
    display: flex;
    gap: 12px;
  }}
  .two-col > div {{
    flex: 1;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Forced Alignment using Montreal Forced Aligner (MFA)</h1>
  <div class="subtitle">IIITH Assignment 2A — Technical Pipeline Summary, Alignment Visualisations &amp; Qualitative Analysis</div>
  <div class="meta-row">
    <span><strong>Student:</strong> [Student Name]</span>
    <span><strong>Roll No.:</strong> [Roll Number]</span>
    <span><strong>Course / Term:</strong> [Course Code / Term]</span>
    <span><strong>Date:</strong> [Submission Date]</span>
  </div>
</div>

<h2>1. Objective</h2>
<p>
  This report documents the setup, execution, and evaluation of a complete automatic forced alignment pipeline using the
  <strong>Montreal Forced Aligner (MFA v3.4.2)</strong> on six provided English speech recordings. Forced alignment takes
  a speech waveform and its known orthographic transcript, maps words to canonical phoneme sequences via a pronunciation
  dictionary, and uses an acoustic model (HMM-GMM/Kaldi in MFA) to estimate word-level and phoneme-level start and end
  timestamps, saved as Praat-compatible <code>.TextGrid</code> files.
</p>

<h2>2. Environment, Models &amp; Custom OOV Pronunciation Handling</h2>
<div class="two-col">
  <div>
    <table>
      <tr><th>Component</th><th>Configuration</th></tr>
      <tr><td>System / CPU</td><td>macOS 27.0.1 (Apple Silicon M2, <code>arm64</code>)</td></tr>
      <tr><td>Environment</td><td>Isolated Conda <code>mfa_env</code> (Python <code>3.10.22</code>)</td></tr>
      <tr><td>MFA Version</td><td><code>3.4.2</code> (<code>conda-forge</code>)</td></tr>
      <tr><td>Acoustic Model</td><td><code>english_us_arpa</code> (<code>v3.0.0a2</code>, ARPAbet)</td></tr>
      <tr><td>Base Dictionary</td><td><code>english_us_arpa</code> (pretrained)</td></tr>
      <tr><td>G2P Model</td><td><code>english_us_arpa</code> (<code>v2.0.0rc4</code>)</td></tr>
      <tr><td>Custom Dictionary</td><td><code>english_us_arpa_custom.dict</code> (+5 OOVs)</td></tr>
    </table>
  </div>
  <div>
    <table>
      <tr><th>OOV Word</th><th>Resolution</th><th>ARPAbet Pronunciation</th></tr>
      <tr><td><code>dukakis</code></td><td>MFA G2P</td><td><code>D AH0 K AA1 K IH0 S</code></td></tr>
      <tr><td><code>maffy</code></td><td>MFA G2P</td><td><code>M AE1 F IY0</code></td></tr>
      <tr><td><code>melnicove</code></td><td>MFA G2P</td><td><code>M EH1 L N IH0 K OW0 V</code></td></tr>
      <tr><td><code>politicize</code></td><td>MFA G2P</td><td><code>P AH0 L IH1 T IH0 S AY2 Z</code></td></tr>
      <tr><td><code>wbur</code></td><td>Manual*</td><td><code>D AH1 B AH0 L Y UW0 B IY1 Y UW1 AA1 R</code></td></tr>
    </table>
    <p style="font-size: 7.7pt; color: #475569;">
      *<code>mfa g2p</code> initially proposed word-like pronunciations (<code>B ER0</code> / <code>W AH0 B ER0</code>) for the radio call sign <code>wbur</code>; it was manually constructed from verified base-dictionary letter entries (W-B-U-R).
    </p>
  </div>
</div>

<h2>3. Dataset Preparation &amp; Alignment Execution</h2>
<p>
  The corpus comprises <strong>6 recordings (97.16 s total)</strong> from two distinct speaker domains:
  <strong><code>F2B</code></strong> (3 broadcast news passages, <code>25.31–30.71 s</code> each, <code>84.66 s</code> total; originally NIST SPHERE format, losslessly converted to RIFF WAV PCM-16 mono 16 kHz) and
  <strong><code>ISLE_SESS0131</code></strong> (3 short minimal-pair contrast utterances, <code>3.88–4.50 s</code> each, <code>12.50 s</code> total; already RIFF WAV PCM-16 mono 16 kHz).
  Original files were preserved untouched and SHA-256 verified. Recordings were grouped into speaker subdirectories
  (<code>alignment_corpus/F2B/</code> and <code>alignment_corpus/ISLE_SESS0131/</code>) for independent speaker normalization.
  Post-OOV validation (<code>mfa validate --ignore_acoustics --clean</code>) confirmed <strong>0 OOVs and 0 errors</strong>.
  Alignment executed cleanly in <code>44.836 s</code> via <code>mfa align alignment_corpus english_us_arpa_custom.dict english_us_arpa aligned_textgrids --clean</code>,
  producing 6 valid <code>.TextGrid</code> files with <strong>0 <code>&lt;unk&gt;</code>, 0 <code>spn</code>, 0 gaps/overlaps, and 0.0000 s duration mismatch</strong>.
</p>

<h2>4. Quantitative Alignment Statistics</h2>
<table>
  <tr>
    <th>Recording</th>
    <th>Group</th>
    <th class="num">Dur (s)</th>
    <th class="num">Words (NE/Tot)</th>
    <th class="num">Phones (NE/Tot)</th>
    <th class="num">Mean/Med Word (ms)</th>
    <th class="num">Mean/Med Phone (ms)</th>
    <th class="num">Min–Max Phone (ms)</th>
    <th class="num">&le;30.01 ms*</th>
    <th class="num">&ge;299.99 ms*</th>
    <th class="num">Rate (wpm)</th>
  </tr>
  <tr>
    <td><code>F2BJRLP1</code></td><td>F2B</td><td class="num">25.31</td><td class="num">70 / 84</td><td class="num">308 / 322</td>
    <td class="num">318.0 / 305.0</td><td class="num">72.3 / 60.0</td><td class="num">10.0 – 230.0</td><td class="num">29*</td><td class="num">0</td><td class="num">166.0</td>
  </tr>
  <tr>
    <td><code>F2BJRLP2</code></td><td>F2B</td><td class="num">28.65</td><td class="num">74 / 84</td><td class="num">345 / 355</td>
    <td class="num">348.5 / 360.0</td><td class="num">74.8 / 70.0</td><td class="num">30.0 – 240.0</td><td class="num">25</td><td class="num">0</td><td class="num">155.0</td>
  </tr>
  <tr>
    <td><code>F2BJRLP3</code></td><td>F2B</td><td class="num">30.71</td><td class="num">82 / 95</td><td class="num">372 / 385</td>
    <td class="num">335.7 / 320.0</td><td class="num">74.0 / 70.0</td><td class="num">30.0 – 290.0</td><td class="num">23</td><td class="num">0</td><td class="num">160.2</td>
  </tr>
  <tr>
    <td><code>ISLE_..._01_sprt1</code></td><td>ISLE</td><td class="num">4.13</td><td class="num">5 / 8</td><td class="num">13 / 16</td>
    <td class="num">332.0 / 390.0</td><td class="num">127.7 / 130.0</td><td class="num">40.0 – 210.0</td><td class="num">0</td><td class="num">0</td><td class="num">72.7</td>
  </tr>
  <tr>
    <td><code>ISLE_..._02_sprt1</code></td><td>ISLE</td><td class="num">3.88</td><td class="num">5 / 8</td><td class="num">12 / 15</td>
    <td class="num">314.0 / 320.0</td><td class="num">130.8 / 100.0</td><td class="num">50.0 – 350.0</td><td class="num">0</td><td class="num">1</td><td class="num">77.4</td>
  </tr>
  <tr>
    <td><code>ISLE_..._03_sprt1</code></td><td>ISLE</td><td class="num">4.50</td><td class="num">5 / 7</td><td class="num">13 / 15</td>
    <td class="num">368.0 / 400.0</td><td class="num">141.5 / 140.0</td><td class="num">40.0 – 260.0</td><td class="num">0</td><td class="num">0</td><td class="num">66.7</td>
  </tr>
  <tr style="font-weight: 600; background: #eff6ff;">
    <td>F2B Aggregate (3)</td><td>F2B</td><td class="num">84.66</td><td class="num">226 / 263</td><td class="num">1,025 / 1,062</td>
    <td class="num">334.4 / 320.0</td><td class="num">73.7 / 70.0</td><td class="num">10.0 – 290.0</td><td class="num">77</td><td class="num">0</td><td class="num">160.2</td>
  </tr>
  <tr style="font-weight: 600; background: #eff6ff;">
    <td>ISLE Aggregate (3)</td><td>ISLE</td><td class="num">12.50</td><td class="num">15 / 23</td><td class="num">38 / 46</td>
    <td class="num">338.0 / 390.0</td><td class="num">133.4 / 115.0</td><td class="num">40.0 – 350.0</td><td class="num">0</td><td class="num">1</td><td class="num">72.0</td>
  </tr>
  <tr style="font-weight: 700; background: #dbeafe;">
    <td>All 6 Recordings</td><td>All</td><td class="num">97.16</td><td class="num">241 / 286</td><td class="num">1,063 / 1,108</td>
    <td class="num">334.7 / 320.0</td><td class="num">75.9 / 70.0</td><td class="num">10.0 – 350.0</td><td class="num">77</td><td class="num">1</td><td class="num">148.8</td>
  </tr>
</table>
<p style="font-size: 7.2pt; color: #475569; margin: -3px 0 4px 0;">
  *Reconciled counts (non-empty phones only; silence excluded) using MFA's &plusmn;0.01 ms frame-grid tolerance for timestamp serialization jitter: <strong>77 nominally 30 ms or shorter (&le;30.01 ms)</strong> = <strong>2 shorter than nominal 30 ms (&lt;29.99 ms</strong>, both 10 ms <code>AH0</code> in <code>F2BJRLP1</code>) + <strong>75 nominally 30 ms (29.99–30.01 ms</strong>, incl. 12 serialized as 30.001 ms; 27/25/23 in <code>F2BJRLP1–3</code>); <strong>1 at least nominally 300 ms (&ge;299.99 ms)</strong>.
</p>

<div class="callout">
  <strong>Inspection Tooling Disclosure:</strong> Praat was not installed on this macOS machine at the time of analysis. All alignment figures below were generated programmatically in Python (<code>praatio</code>, <code>soundfile</code>, <code>scipy</code>, <code>matplotlib</code> at 300 DPI) directly from the unmodified Stage 05 <code>.TextGrid</code> files. Direct interactive Praat GUI and auditory verification remain recommended as a follow-up step.
</div>

<div class="page-break"></div>

<h2>5. Representative Alignment Visualisations — Broadcast News (<code>F2B</code>)</h2>

<div class="figure-box">
  <img src="{FIG_F2B_WBUR}" alt="F2BJRLP1 WBUR's Zoomed Alignment">
  <div class="caption">
    <strong>Figure 1 — <code>F2BJRLP1</code> Zoomed Alignment (20.25–21.75 s): Custom OOV <code>wbur's</code> (20.56–21.44 s).</strong>
    <em>Top panels:</em> time-domain waveform and wideband spectrogram (0–8 kHz, ~8 ms Hann window) with vertical word (solid blue) and phone (dotted green) boundaries.
    <em>Word tier:</em> segments <code>as</code> (200 ms), <code>wbur's</code> (880 ms, highlighted), and <code>margo</code> (300 ms).
    <em>Phone tier:</em> shows all 14 ARPAbet phones of <code>wbur's</code>. While stop closures (<code>D</code>, <code>B</code>), smooth glide/rhotic formants (<code>Y UW1 AA1 R</code>), and final sibilant frication (<code>Z</code>, 4–8 kHz) align cleanly, the forced unstressed schwa <code>AH0</code> at <strong>20.73–20.74 s</strong> (red highlight) is compressed to MFA's single-frame floor (<strong>10 ms</strong>) because fluent broadcast delivery reduces the "-ble-" syllable of "W".
  </div>
</div>

<div class="figure-box">
  <img src="{FIG_F2B_SJCS}" alt="F2BJRLP1 S.J.C.'s Zoomed Alignment">
  <div class="caption">
    <strong>Figure 2 — <code>F2BJRLP1</code> Zoomed Alignment (4.50–5.75 s): Initialism <code>S.J.C.'s</code> (4.79–5.43 s).</strong>
    <em>Word tier:</em> MFA tokenizes the dotted initialism into <code>s</code> (150 ms), <code>j</code> (180 ms), and <code>c's</code> (310 ms) between <code>the</code> and <code>current</code>.
    <em>Phone tier:</em> demonstrates strong acoustic-landmark precision — high-frequency sibilant bands (4–8 kHz) align tightly with <code>S</code> (4.86–4.94 s), <code>S</code> (5.12–5.25 s), and <code>Z</code> (5.35–5.43 s); <code>JH</code> (4.94–5.03 s) captures both closure and postalveolar affricate release; and voiced vowels <code>EH1</code>, <code>EY1</code>, and <code>IY1</code> align cleanly with periodic formant regions.
  </div>
</div>

<div class="page-break"></div>

<h2>6. Representative Alignment Visualisations — Read Speech (<code>ISLE_SESS0131</code>)</h2>

<div class="figure-box">
  <img src="{FIG_ISLE_02}" alt="ISLE 02 Overview Alignment">
  <div class="caption">
    <strong>Figure 3 — <code>ISLE_SESS0131_BLOCKD02_02_sprt1</code> Full Utterance Overview (3.88 s): <em>"I said new not no"</em>.</strong>
    <em>Word tier:</em> identifies all 5 words (<code>i</code> 70 ms, <code>said</code> 320 ms, <code>new</code> 410 ms, <code>not</code> 300 ms, <code>no</code> 470 ms) plus 3 silence intervals (leading, brief pause after <code>new</code>, and trailing).
    <em>Phone tier:</em> shows all 12 non-empty ARPAbet phones. The utterance-final vowel <code>OW1</code> in <code>no</code> (<strong>1.87–2.22 s, 350 ms</strong>, amber highlight) is the longest phone in the corpus; closer inspection reveals that voicing decays to near-silence around ~2.14–2.15 s, indicating ~70–80 ms of right-edge boundary trailing into low-energy decay.
  </div>
</div>

<div class="figure-box">
  <img src="{FIG_ISLE_03}" alt="ISLE 03 Final Stops Zoomed Alignment">
  <div class="caption">
    <strong>Figure 4 — <code>ISLE_SESS0131_BLOCKD02_03_sprt1</code> Zoomed Alignment (0.45–2.40 s): Word-Final Stops in <code>bad</code>, <code>not</code>, <code>bed</code>.</strong>
    <em>Word tier:</em> segments <code>said</code> (370 ms), <code>bad</code> (510 ms), <code>not</code> (400 ms), and <code>bed</code> (490 ms) with no inter-word silence intervals inserted.
    <em>Phone tier:</em> highlights three elongated word-final stops — <code>D</code> in <code>bad</code> (<strong>0.92–1.17 s, 250 ms</strong>), <code>T</code> in <code>not</code> (<strong>1.33–1.57 s, 240 ms</strong>), and <code>D</code> in <code>bed</code> (<strong>1.80–2.06 s, 260 ms</strong>). The spectrogram clearly shows each stop's closure and sharp release burst followed by ~80–110 ms of post-release pause absorbed into the stop interval.
  </div>
</div>

<div class="page-break"></div>

<h2>7. Key Qualitative Observations &amp; Corpus Comparison</h2>

<h3>7.1 <code>F2B</code> (Broadcast News) vs. <code>ISLE_SESS0131</code> (Read Speech)</h3>
<ul>
  <li><strong>Speaking rate &amp; phone duration contrast:</strong> <code>F2B</code> averages <strong>160.2 wpm</strong> with a mean phone duration of <strong>73.7 ms</strong> (median 70.0 ms), whereas <code>ISLE_SESS0131</code> averages <strong>72.0 wpm</strong> (~2.2&times; slower) with a mean phone duration of <strong>133.4 ms</strong> (~1.8&times; longer). Mean word durations appear similar (334.4 ms vs. 338.0 ms) only because <code>F2B</code> contains multisyllabic words (4.54 phones/word) whereas <code>ISLE</code> uses slowly articulated monosyllables (2.53 phones/word).</li>
  <li><strong>Opposite boundary error modes:</strong> Rapid coarticulation in <code>F2B</code> produces <strong>77 phones nominally 30 ms or shorter (&le;30.01 ms; 7.5% of F2B, 7.2% overall: 2 shorter than nominal 30 ms [&lt;29.99 ms] and 75 nominally 30 ms [29.99–30.01 ms])</strong>, risking under-duration or frame-floor compression (e.g., 10 ms <code>AH0</code> in <code>wbur's</code> and <code>will</code>). Conversely, slow emphatic delivery in <code>ISLE_SESS0131</code> produces <strong>0 phones nominally 30 ms or shorter (&le;30.01 ms)</strong> but suffers from right-edge over-extension into low-energy decay (<code>OW1</code> 350 ms, the sole phone at least nominally 300 ms [&ge;299.99 ms]) and absorption of unmodeled inter-word pauses into word-final stops (<code>D</code>/<code>T</code> 240–260 ms).</li>
</ul>

<h3>7.2 Proper-Name &amp; Custom OOV Limitations</h3>
<ul>
  <li><strong>Single-pronunciation rigidity:</strong> Supplying only one canonical entry for <code>wbur</code> forced MFA to emit <code>AH0</code> even when reduced in fast speech. Similarly, <code>mfa g2p</code> assigned <code>AH0</code> to the first syllable of <code>dukakis</code> (<code>D AH0 K AA1 K IH0 S</code>, 2.72–3.36 s in <code>F2BJRLP2</code>) rather than <code>UW0</code>; although major stop and sibilant landmarks aligned cleanly, adding both variants would better model speaker variation.</li>
  <li><strong>Compound &amp; surname handling:</strong> Hyphenated <code>de-politicize</code> (<code>F2BJRLP2</code>, 5.42–6.51 s) and surnames <code>maffy</code> (23.94–24.33 s) and <code>melnicove</code> (<code>F2BJRLP1</code>, 21.74–22.30 s) aligned cleanly at visible spectrogram transitions, though surname phonology cannot be guaranteed without listening verification.</li>
</ul>

<h2>8. Technical Validation vs. Perceptual Accuracy</h2>
<div class="callout-blue">
  <strong>Critical Distinction:</strong> Passing all automated technical checks (<code>0</code> <code>&lt;unk&gt;</code>, <code>0</code> <code>spn</code>, <code>0</code> overlaps/gaps, exact <code>0.0000 s</code> duration match across all 6 TextGrids) confirms <em>structural validity</em>, <strong>not</strong> guaranteed <em>perceptual accuracy</em>. Spectrogram inspection confirms strong landmark alignment on sibilants, stop bursts, and stressed vowels, while exposing micro-level artifacts (10 ms <code>AH0</code> compression, ~70–80 ms vowel trailing, and ~80–110 ms pause absorption into stops). Final perceptual confirmation requires direct auditory playback in Praat.
</div>

<h2>9. Limitations &amp; Potential Improvements</h2>
<ol style="margin: 3px 0 0 0; padding-left: 16px;">
  <li style="margin-bottom: 4px;"><strong>Direct Praat auditory inspection:</strong> Open all 6 <code>.wav</code> + <code>.TextGrid</code> pairs in Praat to listen to flagged intervals (<code>wbur's</code>, <code>dukakis</code>, <code>maffy</code>, <code>melnicove</code>, <code>ISLE 02</code> <code>NO</code>, and <code>ISLE 03</code> final stops) and verify perceptual boundary placement.</li>
  <li style="margin-bottom: 4px;"><strong>Multi-variant custom dictionary:</strong> Add reduced and alternative pronunciations (e.g., <code>wbur</code> without <code>AH0</code>; <code>dukakis</code> with <code>UW0</code>) so MFA can select the best acoustic match during Viterbi alignment.</li>
  <li style="margin-bottom: 4px;"><strong>Alternative acoustic model comparison:</strong> Benchmark <code>english_us_arpa</code> against <code>english_mfa</code> (IPA phone set with finer allophonic modeling of flaps and syllabic consonants) to measure boundary precision gains.</li>
  <li><strong>Silence-probability tuning for slow read speech:</strong> Adjusting optional inter-word silence probabilities or beam settings for slow minimal-pair corpora like <code>ISLE_SESS0131</code> can prevent final stops from absorbing post-release pauses.</li>
</ol>

</body>
</html>
"""

def main():
    with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as tmp:
        tmp.write(HTML)
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            CHROME_BIN,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_OUT}",
            tmp_path.as_uri(),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Generated PDF: {PDF_OUT} ({PDF_OUT.stat().st_size:,} bytes)")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    main()
