---
name: transcribe
description: Turns a link (YouTube, Instagram, TikTok…) or a local audio/video file into MIDI. Downloads the wav, splits stems only when there's more than a solo keyboard, and transcribes with Transkun when it's acoustic piano (piano + voice gets split first, Transkun on the piano, muscriptor on the voice), muscriptor electric_piano on the mix for solo Rhodes/EP, or muscriptor per stem for everything else. Saves everything in $TRANSCRIPTIONS_DIR/<Name> MIDI/ (default ./Transcriptions). Use when the user says "transcribe <link>", "/transcribe", "get the MIDI of this", or pastes a link asking for MIDI or stems.
---

# Transcribe

One command does the whole pipeline:

```bash
transcribe "<link or file>" --mode piano|ep|piano-voice|ep-voice|band [--start 0:30 --end 1:15]
```

(`transcribe` must be on PATH, running `transcribe.py` on a Python that has Transkun,
librosa, pretty_midi and soundfile; see the repo README.) Run it with a long timeout
(10 min); separation is the slow part.

## Step 1: ask what it is, before running anything

Unless the user already said it in the request, ask with AskUserQuestion, one question, header "Material":
- **Piano only** → `--mode piano`: no stem split at all, Transkun straight on the wav. Fastest.
- **Electric piano only** (Rhodes / Wurli / EP) → `--mode ep`: no stem split at all, `muscriptor --instruments electric_piano` straight on the wav.
- **Piano + voice** → `--mode piano-voice`: only the vocals/instrumental split.
- **Electric piano + voice** → `--mode ep-voice`: only the vocals/instrumental split, EP engine on the instrumental.
- **Band / other instruments** → `--mode band`: full 4-stem split + muscriptor per stem.
- **Not sure** → `--mode auto`: 4-stem split first, then it decides.

A solo keyboard (piano or EP alone) never needs stems: use `piano` or `ep`, not `band` with `--skip`. If the answer is "band", pick the engine for the Other stem from what the user said is in it:
- **acoustic piano** (e.g. a piano trio) → `--other transkun`
- **electric piano / Rhodes / Wurli** → `--other electric_piano` (muscriptor restricted to EP). Not Transkun, see below.
- **one other known instrument** → `--other <muscriptor group>`, e.g. `organ`, `clean_electric_guitar`, `synth_pad` (`muscriptor list-instruments`)
- **mixed or unknown** → leave the default (muscriptor unrestricted)

For upright bass, add `--bass acoustic_bass`. When the user doesn't want MIDI for a part (e.g. rap vocals), add `--skip vocals` (repeatable: drums / bass / other / vocals). The stem is still split, but it gets no part file and is left out of Full.mid and All.mid. If any of this is unclear from the title, ask about it in the same AskUserQuestion call rather than in a second round.

`--mode` is required by the script, so it never guesses silently.

## What it does

1. **Audio**: `yt-dlp` bestaudio, converted to `<Name>.wav` (44.1 kHz, 16-bit). Local files are converted the same way.
2. **Tuning**: measures the offset from A440 (librosa on the harmonic part of the mix). Transcribers round to the nearest semitone, so detuned audio, worst near a quarter tone, gets notes landing a semitone off.
   - `--tune auto` (default): if the audio is ≥ 15 cents off, it is shifted back to the grid as `<Name> +47c.wav` (the number is the shift). Stems and MIDI are made from that file, and the original `<Name>.wav` is kept. The shift keeps duration and timing (upward shifts within ~3 ms; downward ones can wobble up to ~20 ms locally, no drift), so the MIDI also lines up with the original wav.
   - `--tune off`: never shift. `--tune 47` / `--tune=-53`: shift by exactly that many cents (when the user gives a number).
   - Near a quarter tone (|offset| ≥ 35 cents) the key is ambiguous: +47 and −53 are a semitone apart. The script prints the other `--tune` value; mention it to the user.
3. **Separation + MIDI by mode**:
   - `piano`: no separation. Transkun on the full mix (notes + CC64 pedal).
   - `ep`: no separation. `muscriptor --instruments electric_piano` on the full mix, part file `EP`.
   - `piano-voice`: `stems -m rofo` (BS-Roformer, ~2× the audio length) writes vocals/instrumental to `stems/`. Transkun on the instrumental, `muscriptor --instruments voice` on the vocals.
   - `ep-voice`: the same split, then `muscriptor --instruments electric_piano` on the instrumental (part file `EP`) and `muscriptor --instruments voice` on the vocals.
   - `band`: `stems -m ft` (htdemucs_ft, ~½ the audio length) writes drums / bass / other / vocals to `stems/`. muscriptor on each stem above 1% energy: drums → `drums`, bass → `electric_bass`, vocals → `voice`, other unrestricted, or as set by `--other` (part file named EP / Piano / Organ / Keys accordingly). `--bass acoustic_bass` for upright bass (only ever one bass group, two duplicate notes).
   - `auto`: the htdemucs_ft split, then from stem energy shares + title/description/tags:
     - drums ≥ 3% or bass ≥ 5% → band
     - other < 5% (e.g. a cappella) → band
     - description names a non-piano instrument and never piano → band
     - vocals ≥ 3% → ep-voice if the description names Rhodes / Wurli / electric piano, else piano-voice
     - description names Rhodes / Wurli / electric piano → ep
     - else → piano
   - **muscriptor timing**: always run with `--detect-tempo false` on the audio with 1 s of silence prepended; the notes are shifted back and anything past the audio's end is dropped. Its tempo detection shifts the whole timeline so its first downbeat sits on a bar line (one beat came out a full bar, 3.61 s, late; another 57 ms late), which reads as "the first beat is missing". Without lead-in it also misses notes in the first ~0.5 s. Never pad the end: a mostly-silent last 5 s chunk gets filled with invented notes.
4. **Output**:
   - `<Name> - Piano/EP/Vocals/Drums/Bass/Other/Keys.mid`, one per part.
   - `<Name> - Full.mid`: every part as its own track, when there is more than one.
   - `<Name> - All.mid`: every non-drum part merged into ONE piano track, so it drops into Live as a single clip. Written whenever there are at least two pitched parts. Same-pitch collisions between parts are cleaned up: duplicates < 30 ms apart become one note, and an overlapping earlier note is shortened. The sustain pedal from Transkun parts is kept.

   Velocity: muscriptor has no dynamics (it writes every note at 100), so its notes are set to a flat 50 (`--velocity N` to change). Transkun parts keep their real per-note dynamics.

   All MIDI files are written at one estimated tempo (drum stem in band mode, else the mix; tempogram peak in 60–150 BPM refined by a line fit through the beat times, then in band mode snapped to the tempo, within ±4%, where the transcribed drum hits best fit a 16th-note grid; 2 decimals) so they line up with the wav when the Live set is at that exact BPM. On solo/rubato piano the number is only nominal. `.transcribe.json` records source, mode, tuning, shift, energies, outputs and the generated audio.

Naming: the YouTube title, minus "(Official Video)" and similar; Instagram → `<uploader> IG <id>`; TikTok → `<uploader> TikTok <id>`. `--start/--end` add a ` 0-30s` suffix. Use `--name` when the automatic name is ugly or the user gave one.

Re-running on the same link reuses the wav and stems. A different `--mode` regenerates only the MIDI and deletes MIDI files that the previous run wrote and this one didn't. A different `--tune` also removes the previous shifted wav and its stems.

## Other details

- **After `auto`**, sanity-check the printed `mode:` line against the title/description. If it's wrong, re-run with the right `--mode`/flags; the wav and stems are reused.
- **Long videos** (> ~10 min, check with `yt-dlp --print duration "<url>"` first if it looks like a long video, live set or tutorial): ask whether to trim with `--start/--end` before running.
- **Instagram login errors** → retry with `--cookies-from-browser chrome`.
- `--model large` for muscriptor when the user asks for higher quality (slower).

Never swap in other transcribers: acoustic piano is always Transkun, everything else muscriptor (no basic-pitch).

**Electric piano is not Transkun's job.** Transkun is trained on acoustic piano. On one Rhodes-type EP stem it caught 42 notes, left 40% of the playing with no notes at all, and matched the stem's harmony at 0.40. `muscriptor --instruments electric_piano` got 98 notes and 0.83 (unrestricted muscriptor 0.76; `-m large` no better). When unsure between engines, transcribe the stem with both into the scratchpad and compare chroma similarity against the stem before choosing.

## Reporting back

Reply briefly:
- the folder path
- the mode and why
- the tuning: in tune, or how many cents off and what shift was applied (plus the alternative near a quarter tone), and which wav to load in Live
- each MIDI file with its note count (including All.mid and which parts it merges)
- the BPM to set in Live

Don't list the stems unless asked.
