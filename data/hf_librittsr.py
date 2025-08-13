from typing import Dict, Any, List
import torch
from datasets import load_dataset, Audio
_PAD = 0
class SimpleCharTokenizer:
    def __init__(self):
        charset = list("abcdefghijklmnopqrstuvwxyz '.,;:!?-()")
        self.stoi = {c:i+1 for i,c in enumerate(charset)}
    def encode(self, text: str):
        text = (text or "").lower()
        ids = [self.stoi.get(ch, self.stoi.get(' ')) for ch in text]
        return torch.tensor(ids, dtype=torch.long)
def load_hf_tts(name, subset, split, streaming, sr, text_field, speaker_field):
    cfg = subset or None
    ds = load_dataset(name, cfg, split=split, streaming=streaming)
    if not streaming:
        try: ds = ds.cast_column("audio", Audio(sampling_rate=sr))
        except Exception: pass
    tok = SimpleCharTokenizer()
    def to_item(ex: Dict[str, Any]):
        audio = ex["audio"]
        wav = torch.tensor(audio["array"]).float()
        if wav.ndim > 1: wav = wav.mean(-1)
        text = ex.get(text_field) or ex.get("text") or ex.get("text_original") or ""
        text_ids = tok.encode(text)
        spk = ex.get(speaker_field, "0")
        return {"text_ids": text_ids, "audio": wav, "speaker_id": str(spk)}
    if streaming:
        def iterator():
            for ex in ds:
                yield to_item(ex)
        return iterator()
    else:
        return [to_item(ex) for ex in ds]
def collate_pad(batch: List[Dict[str, Any]], max_seconds: int, sr: int):
    max_len = int(max_seconds * sr)
    audios = [b["audio"][:max_len] for b in batch]
    texts = [b["text_ids"] for b in batch]
    audio_pad = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    text_pad  = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=_PAD)
    return {"text_ids": text_pad, "audio": audio_pad, "speaker_id": [b["speaker_id"] for b in batch]}
