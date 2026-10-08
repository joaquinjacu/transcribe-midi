#!/usr/bin/env python3
"""transcribe — link (or local audio file) -> wav -> stems -> MIDI.

Everything lands in  $TRANSCRIPTIONS_DIR/<Name> MIDI/   (default ./Transcriptions; --out overrides)

    <Name>.wav                 source audio (44.1 kHz, 16-bit)
    <Name> +47c.wav            retuned to A440 when the source is off (--tune); stems + MIDI use it
    stems/                     htdemucs_ft 4 stems (band/auto) or BS-Roformer vocals/instrumental (piano-voice, ep-voice)
    <Name> - <Part>.mid        one file per part (Piano, EP, Vocals, Bass, Drums, Other / Keys)
    <Name> - Full.mid          all parts together, one track each, when there is more than one
    <Name> - All.mid           every non-drum part in ONE track (a single clip in Live),
                               when there are at least two pitched parts

Modes (--mode is required: say what it is up front so nothing runs that isn't needed)
    piano        Transkun on the full mix (notes + sustain pedal); no stem split
    ep           solo electric piano (Rhodes/Wurli): muscriptor --instruments electric_piano
                 on the full mix; no stem split
    piano-voice  BS-Roformer vocals/instrumental split only; Transkun on the instrumental,
                 muscriptor --instruments voice on the vocals
    ep-voice     same split; muscriptor --instruments electric_piano on the instrumental,
                 muscriptor --instruments voice on the vocals
    band         htdemucs_ft 4 stems, muscriptor on each active one (drums / bass / vocals /
                 other); --other transkun runs Transkun on the Other stem (acoustic piano),
                 --other <muscriptor group> restricts it (e.g. electric_piano for Rhodes/EP)
    auto         only when unsure: 4-stem split, then picks one of the above from stem
                 energies + the video's title/description

Re-running on the same link reuses the wav and stems, so switching mode is cheap.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
import soundfile as sf  # noqa: E402

def find_tool(name, env, *candidates):
    """External command: $<env> if set, else the first executable candidate, else `name` on PATH."""
    if os.environ.get(env):
        return Path(os.environ[env]).expanduser()
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return Path(shutil.which(name) or name)


OUT_ROOT = Path(os.environ.get("TRANSCRIPTIONS_DIR", "Transcriptions")).expanduser().absolute()
# Transkun is usually installed in the same venv as the Python running this script
TRANSKUN = find_tool("transkun", "TRANSKUN", Path(sys.executable).parent / "transkun")
MUSCRIPTOR = find_tool("muscriptor", "MUSCRIPTOR")
STEMS = find_tool("stems", "STEMS_CMD", Path(__file__).resolve().parent / "stems")
MANIFEST = ".transcribe.json"

STEM_NAMES = ["Drums", "Bass", "Other", "Vocals"]
# part names for the Other stem when it's restricted to one muscriptor group
OTHER_PART_NAMES = {"electric_piano": "EP", "acoustic_piano": "Piano", "organ": "Organ",
                    "synth_pad": "Pad", "synth_lead": "Lead"}
# cents off A440 equal temperament below which the audio counts as in tune
TUNE_MIN = 15
# energy share (of the 4-stem total) above which a stem counts as present
ACTIVE = 0.01
PIANO_WORDS = ["piano", "pianist", "keys", "keyboard"]
EP_WORDS = ["rhodes", "wurli", "electric piano", "e-piano", "epiano"]
NON_PIANO_WORDS = ["guitar", "violin", "cello", "viola", "sax", "trumpet", "trombone", "flute",
                   "clarinet", "harp", "ukulele", "synth", "a cappella", "acapella", "a capella",
                   "beatbox", "bass solo", "drum solo"]


def log(msg):
    print(msg, flush=True)


def run(cmd, quiet=True):
    """Run a command; on failure show its output and exit."""
    r = subprocess.run([str(c) for c in cmd], capture_output=quiet, text=True)
    if r.returncode != 0:
        if quiet:
            sys.stderr.write((r.stdout or "") + (r.stderr or ""))
        sys.exit(f"command failed ({r.returncode}): {Path(str(cmd[0])).name}")
    return r


def parse_time(s):
    if s is None:
        return None
    secs = 0.0
    for part in str(s).split(":"):
        secs = secs * 60 + float(part)
    return secs


def fmt_secs(x):
    return f"{x:g}"


# ---------------------------------------------------------------- naming

JUNK = re.compile(r"\s*[\(\[][^\)\]]*\b(official|lyrics?|audio|visuali[sz]er|video|hd|hq|4k)\b[^\)\]]*[\)\]]",
                  re.I)


def clean(s):
    s = JUNK.sub("", s)
    s = re.sub(r'[/\\:*?"<>|\n\r\t]+', " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .-_")
    return s[:80].rstrip(" .-_")


def make_name(info):
    ext = (info.get("extractor_key") or "").lower()
    who = info.get("uploader") or info.get("channel") or info.get("uploader_id") or ""
    vid = info.get("id") or ""
    title = info.get("title") or ""
    if ext.startswith("instagram"):
        return clean(f"{who} IG {vid}")
    if ext.startswith("tiktok"):
        return clean(f"{who} TikTok {vid}")
    if not title or re.match(r"^(video|reel|post) by ", title, re.I):
        return clean(f"{who} - {vid}")
    return clean(title)


# ---------------------------------------------------------------- source audio

def fetch_info(url, cookies):
    cmd = ["yt-dlp", "-J", "--no-playlist", "--no-warnings"]
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    return json.loads(run(cmd + [url]).stdout)


def to_wav(src, dst, start, end):
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)] if start is None else ["-t", str(end - start)]
    cmd += ["-i", src, "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", dst]
    run(cmd)


def download(url, wav, start, end, cookies):
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["yt-dlp", "--no-playlist", "--no-warnings", "-f", "bestaudio/best",
               "-o", f"{tmp}/src.%(ext)s", "--print", "after_move:filepath"]
        if cookies:
            cmd += ["--cookies-from-browser", cookies]
        path = run(cmd + [url]).stdout.strip().splitlines()[-1]
        to_wav(path, wav, start, end)


# ---------------------------------------------------------------- tuning

def measure_tuning(wav):
    """Offset from A440 equal temperament in cents, in [-50, 50)."""
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True, duration=180)
    y = librosa.effects.harmonic(y, margin=2)
    return float(librosa.estimate_tuning(y=y, sr=sr, resolution=0.01)) * 100


def pitch_shift(src, dst, cents):
    """Shift pitch by `cents`, keeping duration and timing.

    ffmpeg here has no rubberband: asetrate+aresample shifts pitch and speed together,
    atempo (WSOLA) undoes the speed. atempo drops a few ms at the start, so the result
    is padded back to the source length. Checked by onset cross-correlation against the
    original: upward shifts stay within ~3 ms, downward ones jitter locally by up to ~20 ms
    (no drift). rubberband would be tighter but isn't installed.
    """
    r = 2 ** (cents / 1200)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "shift.wav"
        run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af",
             f"asetrate=44100*{r:.10f},aresample=44100,atempo={1 / r:.10f}", "-c:a", "pcm_s16le", out])
        x, sr = sf.read(str(out), dtype="int16", always_2d=True)
    n = sf.info(str(src)).frames
    pad = n - len(x)
    x = np.concatenate([np.zeros((pad, x.shape[1]), x.dtype), x]) if pad > 0 else x[-pad:]
    sf.write(str(dst), x, sr, subtype="PCM_16")


# ---------------------------------------------------------------- separation

def stem_file(stem_dir, wav, pattern):
    """Stem of this exact input file (retuned and original stems share the folder)."""
    return next(stem_dir.glob(f"{glob.escape(wav.stem)}_{pattern}"), None)


def split4(wav, stem_dir):
    found = {s: stem_file(stem_dir, wav, f"({s})_htdemucs_ft.flac") for s in STEM_NAMES}
    if all(found.values()):
        log("stems: reusing htdemucs_ft stems")
        return found
    log("stems: htdemucs_ft (4 stems)…")
    run([STEMS, "-m", "ft", "-o", stem_dir, wav])
    found = {s: stem_file(stem_dir, wav, f"({s})_htdemucs_ft.flac") for s in STEM_NAMES}
    missing = [s for s, f in found.items() if f is None]
    if missing:
        sys.exit(f"stems: missing {missing} in {stem_dir}")
    return found


def split_vocals(wav, stem_dir):
    pat = lambda s: stem_file(stem_dir, wav, f"({s})_model_bs_roformer*.flac")  # noqa: E731
    if pat("Vocals") and pat("Instrumental"):
        log("stems: reusing BS-Roformer vocals/instrumental")
    else:
        log("stems: BS-Roformer vocals/instrumental (slow, ~2x the audio length)…")
        run([STEMS, "-m", "rofo", "-o", stem_dir, wav])
    return pat("Vocals"), pat("Instrumental")


def energy_shares(stems):
    e = {}
    for s, f in stems.items():
        x, _ = sf.read(f, dtype="float32")
        e[s] = float(np.mean(np.square(x)))
    tot = sum(e.values()) or 1.0
    return {s: v / tot for s, v in e.items()}


def choose_mode(share, text):
    text = text.lower()
    has_piano = any(w in text for w in PIANO_WORDS)
    non_piano = [w for w in NON_PIANO_WORDS if w in text]
    if share["Drums"] >= 0.03 or share["Bass"] >= 0.05:
        return "band", "drums/bass present"
    if share["Other"] < 0.05:
        return "band", "almost nothing besides vocals (no keys to give Transkun)"
    if non_piano and not has_piano:
        return "band", f"no drums/bass, but the description mentions {', '.join(non_piano)}, not piano"
    ep = [w for w in EP_WORDS if w in text]
    if share["Vocals"] >= 0.03:
        if ep:
            return "ep-voice", f"no drums/bass, vocals present, and the description mentions {', '.join(ep)}"
        return "piano-voice", "no drums/bass, vocals present"
    if ep:
        return "ep", f"no drums/bass/vocals, and the description mentions {', '.join(ep)}"
    return "piano", "no drums/bass/vocals"


# ---------------------------------------------------------------- transcription

# muscriptor needs lead-in: a note in the first ~0.5 s of a file often gets no MIDI at all
# (first bass note of a clip that starts on the downbeat). Transcribing with silence before
# the audio and shifting the notes back fixes it. Not after: muscriptor works in 5 s chunks
# and fills a mostly-silent last chunk with invented notes (150 past the end on one stem).
# Transkun already catches notes at 0.00 s, so it runs unpadded.
PAD = 1.0


def pad_audio(audio, out):
    x, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    z = np.zeros((int(PAD * sr), x.shape[1]), np.float32)
    padded = out.with_suffix(".pad.wav")
    sf.write(str(padded), np.concatenate([z, x]), sr, subtype="PCM_24")
    return padded


def fix_muscriptor_midi(mid, dur, velocity):
    """Shift a MIDI transcribed from padded audio back onto the original timeline, drop
    anything outside the audio, and set the velocity: muscriptor has no dynamics and
    writes every note at 100."""
    pmid = pretty_midi.PrettyMIDI(str(mid))
    for inst in pmid.instruments:
        notes = []
        for n in inst.notes:
            n.start, n.end = max(0.0, n.start - PAD), min(dur, n.end - PAD)
            n.velocity = velocity
            if n.end > n.start:
                notes.append(n)
        inst.notes = notes
        for ev in inst.control_changes + inst.pitch_bends:
            ev.time = max(0.0, ev.time - PAD)
    pmid.write(str(mid))
    return mid


def transkun(audio, out):
    run([TRANSKUN, audio, out])
    return out


def muscriptor(audio, out, model, velocity, instruments=None):
    # --detect-tempo (default best-effort) shifts the whole timeline so its first detected
    # downbeat sits on a bar line: drums came out 3.61 s (one bar) late on one beat, 57 ms
    # on another. Off, times are plain seconds; the tempo is estimated separately below.
    cmd = [MUSCRIPTOR, "transcribe", pad_audio(audio, out), "-o", out, "-m", model,
           "--detect-tempo", "false"]
    if instruments:
        cmd += ["--instruments", instruments]
    run(cmd)
    return fix_muscriptor_midi(out, sf.info(str(audio)).duration, velocity)


def estimate_bpm(audio):
    """Tempo for the MIDI files, so the grid in Live follows the music.

    Plain beat_track jumps between subdivisions on near-identical audio (95.7 vs 65.4 on
    the same beat retuned by 47 cents). Instead: strongest tempogram peak in 60-150 BPM
    as the prior, beat tracking around it, and a straight-line fit through the beat times
    for the exact period.
    """
    import librosa
    y, sr = librosa.load(str(audio), sr=22050, mono=True, duration=240)
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
    bpms = librosa.tempo_frequencies(tg.shape[0], sr=sr)
    ok = (bpms >= 60) & (bpms <= 150)
    if not ok.any() or not np.isfinite(tg).all():
        return 120.0
    prior = float(bpms[ok][np.argmax(tg.mean(axis=1)[ok])])
    _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, start_bpm=prior, tightness=400)
    bpm = prior
    if len(beats) >= 8:
        t = librosa.frames_to_time(beats, sr=sr)
        fit = 60 / np.polyfit(np.arange(len(t)), t, 1)[0]
        # beat_track may still settle on a multiple of the prior; keep the prior's octave
        for k in (0.5, 2 / 3, 1, 1.5, 2):
            if abs(fit * k / prior - 1) < 0.08:
                bpm = fit * k
                break
    while bpm > 150:
        bpm /= 2
    while bpm < 60:
        bpm *= 2
    # Live takes 2 decimals; write exactly what the user will type in
    return round(bpm, 2)


def refine_bpm(bpm, onsets):
    """Tempo within +-4% of `bpm` whose 16th-note grid best fits the drum onsets.

    The beat-time fit drifts on short clips (99.67 where the drum hits sit on 98.00 in a
    16 s clip); transcribed drum hits pin the grid down to ~0.05 BPM.
    """
    t = np.asarray(sorted(onsets))
    if len(t) < 16:
        return bpm
    cands = np.arange(bpm * 0.96, bpm * 1.04, 0.01)
    best, best_score = bpm, -1.0
    for b in cands:
        step = 60 / b / 4
        phases = np.linspace(0, step, 32, endpoint=False)[:, None]
        r = (t[None, :] - phases) / step
        err = np.abs(r - np.round(r)) * step
        score = np.exp(-(err / 0.012) ** 2).mean(axis=1).max()
        if score > best_score:
            best, best_score = b, score
    return round(float(best), 2)


def load_part(path, part):
    """Instruments of a raw MIDI, renamed after the part."""
    insts = [i for i in pretty_midi.PrettyMIDI(str(path)).instruments if i.notes]
    for i in insts:
        raw = i.name.strip()
        i.name = part if len(insts) == 1 or not raw else f"{part} - {raw.title()}"
    return insts


def merge_pitched(insts):
    """Every non-drum note in one piano track, so it drops into Live as a single clip.

    On one channel, same-pitch notes from different parts would cut each other off:
    near-simultaneous duplicates (< 30 ms apart) become one note, and an earlier note
    that overlaps a later one is shortened to end where the later one starts.
    """
    notes = sorted((pretty_midi.Note(n.velocity, n.pitch, n.start, n.end)
                    for i in insts if not i.is_drum for n in i.notes),
                   key=lambda n: (n.pitch, n.start))
    merged = []
    for n in notes:
        prev = merged[-1] if merged and merged[-1].pitch == n.pitch else None
        if prev and n.start < prev.end:
            if n.start - prev.start < 0.03:
                prev.end = max(prev.end, n.end)
                prev.velocity = max(prev.velocity, n.velocity)
                continue
            prev.end = n.start
        merged.append(n)
    out = pretty_midi.Instrument(program=0, name="All")
    out.notes = sorted((n for n in merged if n.end - n.start > 0.005), key=lambda n: (n.start, n.pitch))
    # keep the sustain pedal (Transkun parts) so the piano part still rings
    out.control_changes = sorted((pretty_midi.ControlChange(c.number, c.value, c.time)
                                  for i in insts if not i.is_drum for c in i.control_changes
                                  if c.number == 64), key=lambda c: c.time)
    return out


def write_midi(insts, path, bpm):
    pm = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=bpm)
    pm.instruments.extend(insts)
    pm.write(str(path))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="URL (YouTube, Instagram, TikTok, …) or a local audio/video file")
    ap.add_argument("--mode", choices=["piano", "ep", "piano-voice", "ep-voice", "band", "auto"], required=True,
                    help="piano and ep skip separation, piano-voice and ep-voice skip the 4-stem split; "
                    "auto only when unsure")
    ap.add_argument("--name", help="folder/file base name (default: from the video title)")
    ap.add_argument("--start", help="trim start, seconds or m:ss")
    ap.add_argument("--end", help="trim end, seconds or m:ss")
    ap.add_argument("--other", "--keys", dest="other", default="muscriptor",
                    help="band mode, Other stem: muscriptor (unrestricted, default), transkun "
                    "(acoustic piano), or a muscriptor group to restrict to, e.g. electric_piano, "
                    "organ, clean_electric_guitar (see `muscriptor list-instruments`)")
    ap.add_argument("--bass", default="electric_bass", help="band mode: muscriptor bass group "
                    "(electric_bass or acoustic_bass; only one, two groups duplicate notes)")
    ap.add_argument("--skip", action="append", default=[], choices=["drums", "bass", "other", "vocals"],
                    help="don't transcribe this stem (repeatable), e.g. --skip vocals for a rap record; "
                    "it is left out of Full.mid and All.mid too")
    ap.add_argument("--velocity", type=int, default=50,
                    help="velocity for muscriptor notes (it has no dynamics); Transkun keeps its own")
    ap.add_argument("--model", default="medium", choices=["small", "medium", "large"],
                    help="muscriptor model size")
    ap.add_argument("--tune", default="auto",
                    help="auto (default): retune to A440 when the audio is >= 15 cents off; off: never; "
                    "or a number = cents to shift the audio by, e.g. 47 or -53 (use --tune=-53)")
    ap.add_argument("--cookies-from-browser", dest="cookies", help="pass to yt-dlp, e.g. chrome")
    ap.add_argument("--out", default=str(OUT_ROOT),
                    help="parent folder (default: $TRANSCRIPTIONS_DIR, else ./Transcriptions)")
    a = ap.parse_args()

    start, end = parse_time(a.start), parse_time(a.end)
    local = Path(a.source).expanduser()
    if local.exists():
        info = {"title": local.stem}
        base = a.name or clean(local.stem)
    else:
        info = fetch_info(a.source, a.cookies)
        base = a.name or make_name(info)
    if start is not None or end is not None:
        base += f" {fmt_secs(start or 0)}-{fmt_secs(end) if end is not None else 'end'}s"

    folder = Path(a.out) / f"{base} MIDI"
    folder.mkdir(parents=True, exist_ok=True)
    wav = folder / f"{base}.wav"
    log(f"folder: {folder}")

    if wav.exists():
        log("audio: reusing existing wav")
    elif local.exists():
        log("audio: converting local file…")
        to_wav(str(local), wav, start, end)
    else:
        log("audio: downloading…")
        download(a.source, wav, start, end, a.cookies)
    dur = sf.info(str(wav)).duration
    log(f"audio: {dur:.1f}s")

    # tuning: transcribers round to the nearest semitone, so audio far off A440 (worst
    # near a quarter tone) gets notes landing a semitone apart. Retune before anything else.
    offset = measure_tuning(wav)
    log(f"tuning: {offset:+.0f} cents from A440 (A4 = {440 * 2 ** (offset / 1200):.1f} Hz)")
    if a.tune == "off":
        shift = 0
    elif a.tune == "auto":
        shift = -round(offset) if abs(offset) >= TUNE_MIN else 0
    else:
        shift = round(float(a.tune))
    work = wav
    if shift:
        work = folder / f"{base} {shift:+d}c.wav"
        if work.exists():
            log(f"tuning: reusing {work.name}")
        else:
            pitch_shift(wav, work, shift)
        log(f"tuning: audio shifted {shift:+d} cents -> {work.name}, now {measure_tuning(work):+.0f} cents")
        if abs(offset) >= 35:
            alt = shift - 100 if shift > 0 else shift + 100
            log(f"tuning: near a quarter tone, so the key is ambiguous; --tune={alt:+d} gives the "
                f"version a semitone {'lower' if alt < shift else 'higher'}")
    elif abs(offset) >= TUNE_MIN:
        log("tuning: left as is (--tune off); notes may round to the wrong semitone")

    mode, why = a.mode, "requested"
    stems = share = None
    if mode in ("band", "auto"):
        stems = split4(work, folder / "stems")
        share = energy_shares(stems)
        log("stem energy: " + "  ".join(f"{s} {share[s]:.0%}" for s in STEM_NAMES))
    if mode == "auto":
        text = " ".join(str(info.get(k) or "") for k in ("title", "description", "uploader"))
        text += " " + " ".join(info.get("tags") or [])
        mode, why = choose_mode(share, text)
    log(f"mode: {mode} ({why})")

    parts = []  # (part name, [instruments])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if mode == "piano":
            log("midi: Transkun on the mix…")
            parts.append(("Piano", load_part(transkun(work, tmp / "piano.mid"), "Piano")))
        elif mode == "ep":
            # Transkun is trained on acoustic piano and misses much of a Rhodes/Wurli
            log("midi: muscriptor (electric_piano) on the mix…")
            parts.append(("EP", load_part(muscriptor(work, tmp / "ep.mid", a.model, a.velocity, "electric_piano"), "EP")))
        elif mode in ("piano-voice", "ep-voice"):
            vocals, inst = split_vocals(work, folder / "stems")
            if mode == "piano-voice":
                log("midi: Transkun on the instrumental…")
                parts.append(("Piano", load_part(transkun(inst, tmp / "piano.mid"), "Piano")))
            else:
                log("midi: muscriptor (electric_piano) on the instrumental…")
                parts.append(("EP", load_part(muscriptor(inst, tmp / "ep.mid", a.model, a.velocity, "electric_piano"), "EP")))
            if "vocals" in a.skip:
                log("midi: skipping Vocals (--skip)")
            else:
                log("midi: muscriptor (voice) on the vocals…")
                parts.append(("Vocals", load_part(muscriptor(vocals, tmp / "vocals.mid", a.model, a.velocity, "voice"), "Vocals")))
        else:
            jobs = [("Drums", "drums"), ("Bass", a.bass), ("Other", None), ("Vocals", "voice")]
            for stem, group in jobs:
                if stem.lower() in a.skip:
                    log(f"midi: skipping {stem} (--skip)")
                    continue
                if share[stem] < ACTIVE:
                    log(f"midi: skipping {stem} (silent, {share[stem]:.1%})")
                    continue
                out = tmp / f"{stem.lower()}.mid"
                part = stem
                if stem == "Other" and a.other == "transkun":
                    log("midi: Transkun on the Other stem…")
                    parts.append(("Keys", load_part(transkun(stems[stem], out), "Keys")))
                    continue
                if stem == "Other" and a.other != "muscriptor":
                    group = a.other
                    part = OTHER_PART_NAMES.get(group, group.replace("_", " ").title())
                log(f"midi: muscriptor on {stem}" + (f" ({group})" if group else "") + "…")
                parts.append((part, load_part(muscriptor(stems[stem], out, a.model, a.velocity, group), part)))

    parts = [(p, insts) for p, insts in parts if insts]
    if not parts:
        sys.exit("no notes found in any part")

    bpm = estimate_bpm(stems["Drums"] if mode == "band" and share["Drums"] >= 0.03 else work)
    drum_onsets = [n.start for p, insts in parts if p == "Drums" for i in insts for n in i.notes]
    if drum_onsets:
        bpm = refine_bpm(bpm, drum_onsets)
    outputs = []
    for part, insts in parts:
        path = folder / f"{base} - {part}.mid"
        write_midi(insts, path, bpm)
        outputs.append(path.name)
    if len(parts) > 1:
        path = folder / f"{base} - Full.mid"
        write_midi([i for _, insts in parts for i in insts], path, bpm)
        outputs.append(path.name)
    pitched = [i for _, insts in parts for i in insts if not i.is_drum]
    all_inst = None
    if len(pitched) > 1:
        all_inst = merge_pitched(pitched)
        path = folder / f"{base} - All.mid"
        write_midi([all_inst], path, bpm)
        outputs.append(path.name)

    # drop MIDI files a previous run (other mode) wrote that this run did not regenerate
    stem_dir = folder / "stems"
    audio = ([work.name] if work != wav else []) + sorted(
        f"stems/{f.name}" for f in stem_dir.glob(f"{glob.escape(work.stem)}_(*") if stem_dir.exists())
    man = folder / MANIFEST
    if man.exists():
        prev = json.loads(man.read_text())
        for old in prev.get("outputs", []) + prev.get("audio", []):
            if old not in outputs + audio and (folder / old).exists():
                (folder / old).unlink()
                log(f"removed stale {old}")
    man.write_text(json.dumps({
        "source": a.source if not local.exists() else str(local), "name": base, "mode": mode,
        "reason": why, "tuning_cents": round(offset, 1), "shift_cents": shift, "bpm": bpm, "stem_energy": share and {k: round(v, 4) for k, v in share.items()},
        "muscriptor_model": a.model, "outputs": outputs, "audio": audio}, indent=2, ensure_ascii=False))

    log(f"\ntempo written into the MIDI: {bpm:.2f} BPM (approximate; set the Live set to exactly this "
        "and the MIDI lines up with the wav)")
    for part, insts in parts:
        n = sum(len(i.notes) for i in insts)
        log(f"  {base} - {part}.mid  {n} notes" + (f"  [{', '.join(i.name for i in insts)}]" if len(insts) > 1 else ""))
    if len(parts) > 1:
        log(f"  {base} - Full.mid  all parts")
    if all_inst:
        log(f"  {base} - All.mid  {len(all_inst.notes)} notes, one clip: "
            f"{' + '.join(i.name for i in pitched)} (no drums)")
    if shift:
        log(f"  audio retuned {shift:+d} cents: use {work.name} in Live (or detune the original "
            f"{shift:+d} c); the MIDI and stems match that")
    log(f"done: {folder}")


if __name__ == "__main__":
    main()
