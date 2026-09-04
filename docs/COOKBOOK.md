# AuK Inference Cookbook

AuK uses a unified ChatML-style input format, with each task specified by a natural-language instruction and audio supplied when required. This guide provides English and Chinese instruction templates, along with CLI and Python examples.

## Contents

- [Python API Setup](#python-api-setup)
- [1. Speech Generation](#1-speech-generation)
  - [1.1 Zero-shot TTS](#11-zero-shot-tts)
  - [1.2 Instruct TTS](#12-instruct-tts)
- [2. Content Editing](#2-content-editing)
  - [2.1 Speech Content Editing](#21-speech-content-editing)
  - [2.2 Lyric Editing](#22-lyric-editing)
- [3. Acoustic Editing](#3-acoustic-editing)
  - [3.1 Pitch Editing](#31-pitch-editing)
  - [3.2 Speed Editing](#32-speed-editing)
  - [3.3 Volume Editing](#33-volume-editing)
- [4. Paralinguistic Editing](#4-paralinguistic-editing)
  - [4.1 Emotion](#41-emotion)
  - [4.2 Timbre](#42-timbre)
  - [4.3 De-accent](#43-de-accent)
  - [4.4 Nonverbal Editing](#44-nonverbal-editing)
  - [4.5 Whisper Conversion](#45-whisper-conversion)
- [5. Enhancement & Separation](#5-enhancement--separation)
  - [5.1 Speech Enhancement](#51-speech-enhancement)
  - [5.2 Speech Separation](#52-speech-separation)
  - [5.3 Music Separation](#53-music-separation)
  - [5.4 Target Speaker Extraction](#54-target-speaker-extraction)

---

## Python API Setup

Initialize AuK once and reuse `run_auk(...)` in the examples below. Leave `audio_path` unset for text-only Instruct TTS.

```python
from auk.infer.infer_auk import AukInfer, save_audio

checkpoint = "ckpts/AuK/auk_base.safetensors"
config = "ckpts/AuK/config.yaml"

# Use AuK-Flash instead:
# checkpoint = "ckpts/AuK-Flash/auk_flash.safetensors"
# config = "ckpts/AuK-Flash/config.yaml"

engine = AukInfer(
    config,
    checkpoint,
)


def run_auk(
    instruction,
    output_path,
    audio_path=None,
    gen_seconds=None,
):
    content = [{"type": "text", "text": instruction}]

    if audio_path is not None:
        content.append({"type": "audio", "audio": audio_path})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    audio, sr = engine.generate(
        messages,
        gen_seconds=gen_seconds,
    )
    save_audio(audio, sr, output_path)
```

Pass `gen_seconds` when the task requires an explicit output duration, such as TTS, content editing, or speed editing. For tasks that preserve the source duration, it can be omitted.

## 1. Speech Generation

### 1.1 Zero-shot TTS

Speak the target text in the voice of the reference audio.

**Template**
- EN & CN: Say the following with the same voice: "{text}"

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/zero-shot-tts/ref.wav \
    --instruction "Say the following with the same voice: 'Ladies and gentlemen, it's an honor to have the opportunity to address such a distinguished audience'" \
    --output out_zeroshot_tts.wav \
    --gen_seconds 6.0
```

**Python API example**
```python
run_auk(
    "Say the following with the same voice: 'Ladies and gentlemen, it's an honor to have the opportunity to address such a distinguished audience'",
    "out_zeroshot_tts.wav",
    audio_path="assets/demo-input-audio/zero-shot-tts/ref.wav",
    gen_seconds=6.0,
)
```

### 1.2 Instruct TTS

Generate speech from a voice description alone — no reference audio.

**Template**
- EN: Generate speech based on the following description: "{voice description}". The content to speak is: "{text}".
- CN: 请基于下面的描述: "{声音描述}",生成语音内容"{文本}".

**CLI example**

> Instruct TTS has no reference audio, so omit `--audio` and set a target length with `--gen_seconds`.

```bash
auk-infer \
    --instruction 'Based on the following description: "一位二十多岁的女性，面对刚到家的伴侣，以温柔、关心且略带撒娇的语气轻声诉说。她的声音柔和亲密，语速稍缓，音量适中，音色甜美自然，在短语结尾处语调温暖地上扬。她正向伴侣问候回家，并温柔询问今日工作如何。", generate speech content "Welcome home, how was work today?".' \
    --output out_instruct_tts.wav \
    --gen_seconds 1.7
```

**Python API example**
```python
instruction = (
    'Based on the following description: '
    '"一位二十多岁的女性，面对刚到家的伴侣，以温柔、关心且略带撒娇的语气轻声诉说。'
    '她的声音柔和亲密，语速稍缓，音量适中，音色甜美自然，在短语结尾处语调温暖地上扬。'
    '她正向伴侣问候回家，并温柔询问今日工作如何。", '
    'generate speech content "Welcome home, how was work today?".'
)
# No reference audio -> leave audio_path=None; you must set a length (gen_seconds, or gen_text estimate).
run_auk(instruction, "out_instruct_tts.wav", gen_seconds=1.7)
```

---

## 2. Content Editing

### 2.1 Speech Content Editing

Rewrite *what is said*: replace / insert / remove.

**Template**
- Replace
  - EN: Replace '{original}' with '{new}'.
  - CN: 把‘{原文}’改成‘{新文}’
- Insert
  - EN: Add '{text}' before '{anchor}'. | Add '{text}' after '{anchor}'.
  - CN: 在‘{锚点}’前面加上‘{内容}’ | 在‘{锚点}’后面加上‘{内容}’
- Remove
  - EN: Remove '{text}'. | Remove '{text}' before/after '{anchor}'.
  - CN: 删掉‘{内容}’ | 删掉‘{锚点}’前/后面的‘{内容}’

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/content-edit/content.wav \
    --instruction "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'." \
    --output out_content_edit.wav \
    --gen_seconds 7.0
```

**Python API example**
```python
run_auk(
    "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.",
    "out_content_edit.wav",
    audio_path="assets/demo-input-audio/content-edit/content.wav",
    gen_seconds=7.0,
)
```

### 2.2 Lyric Editing

Rewrite lyrics in a singing recording while preserving the melody and voice.

> **Input must be an a cappella (isolated vocals) recording** — clean solo singing
> with no instrumental backing / background music. If your track has accompaniment,
> extract the vocals first (e.g. via [Music Separation](#53-music-separation)).

**Template**
- EN: Change "{original lyrics}" to "{new lyrics}" in the vocal recording.
- CN: 把这段歌词中的“{原歌词}”改成“{新歌词}”。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/vocal-edit/vocaledit-en-1-input.wav \
    --instruction 'Change "rear view" to "like you" in the vocal recording.' \
    --output out_lyric_edit.wav
```

**Python API example**
```python
run_auk(
    'Change "rear view" to "like you" in the vocal recording.',
    "out_lyric_edit.wav",
    audio_path="assets/demo-input-audio/vocal-edit/vocaledit-en-1-input.wav",
)
```

---

## 3. Acoustic Editing

### 3.1 Pitch Editing

Raise/lower the pitch by semitones; same-length output.

**Template**
- EN: Raise the pitch by {1/2/3} semitones. | Lower the pitch by {1/2/3} semitones.
- CN: 将音调升高{1/2/3}个半音。 | 将音调降低{1/2/3}个半音。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/pitch/pitch-1-input.wav \
    --instruction "Raise the pitch by 2 semitones." \
    --output out_pitch.wav
```

**Python API example**
```python
run_auk(
    "Raise the pitch by 2 semitones.",
    "out_pitch.wav",
    audio_path="assets/demo-input-audio/pitch/pitch-1-input.wav",
)
```

### 3.2 Speed Editing

Adjust speaking rate; output length scales with the speed factor.

**Template**
- EN: Adjust the speech speed to {0.5/0.75/1.25/1.5/2.0}x.
- CN: 将语速调整为{0.5/0.75/1.25/1.5/2.0}倍。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/speed/speed-edit-1-input.wav \
    --instruction "Adjust the speech speed to 1.5x." \
    --output out_speed.wav \
    --gen_seconds 6.86   # speed changes duration; set the expected target length
```

**Python API example**
```python
run_auk(
    "Adjust the speech speed to 1.5x.",
    "out_speed.wav",
    audio_path="assets/demo-input-audio/speed/speed-edit-1-input.wav",
    gen_seconds=6.86,
)
```

### 3.3 Volume Editing

Raise/lower the volume by decibels; same-length output.

**Template**
- EN: Increase the volume by {5/10/15} dB. | Decrease the volume by {5/10/15} dB.
- CN: 将音量升高{5/10/15}分贝。 | 将音量降低{5/10/15}分贝。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/energy/energy-edit-1-input.wav \
    --instruction "Increase the volume by 10 dB." \
    --output out_volume.wav
```

**Python API example**
```python
run_auk(
    "Increase the volume by 10 dB.",
    "out_volume.wav",
    audio_path="assets/demo-input-audio/energy/energy-edit-1-input.wav",
)
```

---

## 4. Paralinguistic Editing

### 4.1 Emotion

Change the emotion while preserving content and voice; same-length output.

**Template**
- EN: Change the emotion to {happy/angry/sad/fearful/surprised/disgusted/calm/excited}.
- CN: 将情感转变为{开心/愤怒/悲伤/恐惧/惊讶/厌恶/平静/兴奋}。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/emotion-edit/en-1-input.wav \
    --instruction "Change the emotion to happy." \
    --output out_emotion.wav
```

**Python API example**
```python
run_auk(
    "Change the emotion to happy.",
    "out_emotion.wav",
    audio_path="assets/demo-input-audio/emotion-edit/en-1-input.wav",
)
```

### 4.2 Timbre

Keep the spoken content unchanged and change the timbre to a description.

**Template**
- EN: Keep the spoken content unchanged and change the timbre to: "{description}".
- CN: 请将这段音频的音色修改为符合以下描述的声音：“{音色描述}”。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/vc/vc-1-input.wav \
    --instruction 'Keep the spoken content unchanged and change the timbre to: "a deep, calm male voice".' \
    --output out_timbre.wav
```

**Python API example**
```python
run_auk(
    'Keep the spoken content unchanged and change the timbre to: "a deep, calm male voice".',
    "out_timbre.wav",
    audio_path="assets/demo-input-audio/vc/vc-1-input.wav",
)
```

### 4.3 De-accent

Remove a regional accent while preserving the speaker's voice and content; same-length output.

**Template**
- EN: Remove the regional accent while preserving the speaker's voice and content.
- CN: 请去掉这段语音里的方言口音，保持说话人音色一致。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/accent/accent-sichuan-input.wav \
    --instruction "请把方言腔改成标准普通话发音,音色维持一致。" \
    --output out_deaccent.wav
```

**Python API example**
```python
run_auk(
    "请把方言腔改成标准普通话发音,音色维持一致。",
    "out_deaccent.wav",
    audio_path="assets/demo-input-audio/accent/accent-sichuan-input.wav",
)
```

### 4.4 Nonverbal Editing

Remove or add nonverbal sounds such as breaths, laughs, or coughs.

**Template**
- Remove
  - EN: Remove all {breaths/laughs/coughs/etc.} from the audio.
  - CN: 删除音频中所有的{换气声/笑声/咳嗽声等}。
- Add
  - EN: Add a {sound} at the {beginning/end} of the speech.
  - CN: 在语音{开头/结尾}增加{声音}。

**CLI example**
```bash
# Remove (set gen_seconds for the target length)
auk-infer \
    --audio assets/demo-input-audio/nv/en-d-input.wav \
    --instruction "Remove the humming from the audio." \
    --output out_nonverbal_remove.wav \
    --gen_seconds 22.0

# Add (gets longer; set gen_seconds)
auk-infer \
    --audio assets/demo-input-audio/nv/en-c-input.wav \
    --instruction "Add a cough before 'We tested'" \
    --output out_nonverbal_add.wav \
    --gen_seconds 10.44
```

**Python API example**
```python
run_auk(
    "Remove the humming from the audio.",
    "out_nonverbal_remove.wav",
    audio_path="assets/demo-input-audio/nv/en-d-input.wav",
    gen_seconds=22.0,
)
# "add" tasks get longer, so pass gen_seconds:
run_auk(
    "Add a cough before 'We tested'",
    "out_nonverbal_add.wav",
    audio_path="assets/demo-input-audio/nv/en-c-input.wav",
    gen_seconds=10.44,
)
```

### 4.5 Whisper Conversion

Convert normal speech ↔ whisper while preserving speaker and content; same-length output.

**Template**
- To whisper
  - EN: Convert this speech into a soft whisper while preserving the speaker and content.
  - CN: 用小声耳语的方式把这段话说出来。
- From whisper
  - EN: Convert this whispered speech into a normal speaking voice while preserving the speaker and content.
  - CN: 把这段耳语转换成正常说话的声音。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/whisper/wh-w2n-zh-input.wav \
    --instruction "用小声耳语的方式把这段话说出来。" \
    --output out_whisper.wav
```

**Python API example**
```python
run_auk(
    "用小声耳语的方式把这段话说出来。",
    "out_whisper.wav",
    audio_path="assets/demo-input-audio/whisper/wh-w2n-zh-input.wav",
)
```

---

## 5. Enhancement & Separation

> All tasks here produce same-length output, so the Python examples omit `gen_seconds`.

### 5.1 Speech Enhancement

Denoise / dereverberate / full enhancement / quality restoration.

**Template**
- Denoise
  - EN: Remove only the background noise, preserve everything else, and output audio of the same length.
  - CN: 请只去除背景噪声，保留其他内容，输出等长结果。
- Dereverberate
  - EN: Remove only the room reverberation, preserve everything else, and output audio of the same length.
  - CN: 请只去除房间混响，保留其他内容，输出等长结果。
- Enhance speech
  - EN: Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.
  - CN: 请保留所有说话人，去除噪声和混响，输出等长的纯净语音。
- Quality restoration
  - EN: Repair the {telephone effect/muffling/clipping/dropouts} and restore natural, clear speech.
  - CN: 请修复这段音频的{电话感/闷声/削波/丢包}，恢复自然清晰的人声。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/se/se-zh-1-input.wav \
    --instruction "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length." \
    --output out_enhance.wav
```

**Python API example**
```python
run_auk(
    "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.",
    "out_enhance.wav",
    audio_path="assets/demo-input-audio/se/se-zh-1-input.wav",
)
```

### 5.2 Speech Separation

Keep one speaker by talking order and remove the others.

**Template**
- EN: Keep only the {first/second/etc.} speaker to start talking and remove all other speakers.
- CN: 只保留第{序号}个开始说话的人，去掉其余说话人。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/ss/zh-1-input.wav \
    --instruction "Please keep the second speaker to start talking and remove the other speakers, outputting a single clean speech track." \
    --output out_separation.wav
```

**Python API example**
```python
run_auk(
    "Please keep the second speaker to start talking and remove the other speakers, outputting a single clean speech track.",
    "out_separation.wav",
    audio_path="assets/demo-input-audio/ss/zh-1-input.wav",
)
```

### 5.3 Music Separation

Extract the singing voice from a mix, or keep all human voices.

**Template**
- Singing only
  - EN: Keep only the singing voice and remove everything else.
  - CN: 请只保留歌声，其余声音都去掉。
- All human voices
  - EN: Keep all human voices, including speech and singing, and remove everything else.
  - CN: 请保留所有人声，包括说话和歌唱，其余声音都去掉。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/vocal-extraction/vocal-1-input.wav \
    --instruction "Keep the clean singing voice, drop all other audio." \
    --output out_music_sep.wav
```

**Python API example**
```python
run_auk(
    "Keep the clean singing voice, drop all other audio.",
    "out_music_sep.wav",
    audio_path="assets/demo-input-audio/vocal-extraction/vocal-1-input.wav",
)
```

### 5.4 Target Speaker Extraction

Locate and keep the target speaker by *what they say*, removing the others.

**Template**
- EN: Keep only the speaker who says "{content}" and remove all other speakers.
- CN: 请只保留说“{内容}”的人，去掉其他说话人。

**CLI example**
```bash
auk-infer \
    --audio assets/demo-input-audio/ss/en-1-input.wav \
    --instruction 'Keep only the speaker who says "get what" and remove all other speakers.' \
    --output out_tse.wav
```

**Python API example**
```python
run_auk(
    'Keep only the speaker who says "get what" and remove all other speakers.',
    "out_tse.wav",
    audio_path="assets/demo-input-audio/ss/en-1-input.wav",
)
```
