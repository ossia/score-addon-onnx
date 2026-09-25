# Generates the fixtures of test_tokenizer.cpp: two tiny GPT-2 style byte-level
# BPE tokenizer.json files. Byte b is token b; "merged" also merges "hello"
# (259) and " world" (264), "bytes" has no merges.
import json
import os

here = os.path.dirname(os.path.abspath(__file__))

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

def save(name, merges):
    u = bytes_to_unicode()
    vocab = {u[b]: b for b in range(256)}
    for m in merges:
        vocab[m.replace(" ", "")] = len(vocab)
    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{
            "id": len(vocab), "content": "<|endoftext|>", "single_word": False,
            "lstrip": False, "rstrip": False, "normalized": False, "special": True}],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
            "use_regex": True},
        "post_processor": {
            "type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False,
            "use_regex": True},
        "decoder": {
            "type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True,
            "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": False, "byte_fallback": False, "vocab": vocab,
            "merges": merges},
    }
    os.makedirs(os.path.join(here, name), exist_ok=True)
    with open(os.path.join(here, name, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, separators=(",", ":"))

save("merged", ["h e", "l l", "he ll", "hell o", "Ġ w", "o r", "Ġw or", "l d", "Ġwor ld"])
save("bytes", [])
