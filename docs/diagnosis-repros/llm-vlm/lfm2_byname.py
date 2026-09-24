# Prototype of by-name binding for LFM2 (conv state + num_logits_to_keep).
import json, re, numpy as np, onnxruntime as ort
try: import regex as rx
except ImportError: rx=None
D="/mnt/win2/models/LFM2-1.2B"
t=json.load(open(D+"/tokenizer.json")); vocab=t['model']['vocab']; inv={v:k for k,v in vocab.items()}
for a in t['added_tokens']: inv[a['id']]=a['content']; vocab[a['content']]=a['id']
ranks={tuple(m if isinstance(m,list) else m.split(' ')):i for i,m in enumerate(t['model']['merges'])}
bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1)); cs=bs[:]; n=0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256+n); n+=1
b2u={b:chr(c) for b,c in zip(bs,cs)}; u2b={v:k for k,v in b2u.items()}
def bpe(w):
    w=list(w)
    while len(w)>1:
        pairs=[(ranks.get((w[i],w[i+1]),1e18),i) for i in range(len(w)-1)]; r,i=min(pairs)
        if r==1e18: break
        w=w[:i]+[w[i]+w[i+1]]+w[i+2:]
    return w
pat=t['pre_tokenizer']['pretokenizers'][0]['pattern']['Regex']
def enc_text(s):
    out=[]
    for piece in rx.findall(pat,s):
        u=''.join(b2u[b] for b in piece.encode()); out+= [vocab[x] for x in bpe(u)]
    return out
def encode(s):
    specials=sorted([a['content'] for a in t['added_tokens']],key=len,reverse=True)
    ids=[]; 
    parts=re.split("("+"|".join(map(re.escape,specials))+")",s)
    for p in parts:
        if not p: continue
        ids+= [vocab[p]] if p in specials else enc_text(p)
    return ids
def decode(ids): return bytes(u2b.get(c,ord('?')) for i in ids for c in inv[i] if i in inv).decode(errors='replace')
prompt="<|startoftext|><|im_start|>user\nWhat is the capital of France? Answer briefly.<|im_end|>\n<|im_start|>assistant\n"
ids=encode(prompt)
so=ort.SessionOptions(); so.log_severity_level=3
s=ort.InferenceSession(D+"/onnx/model_q4.onnx",so,providers=["CPUExecutionProvider"])
TY={"tensor(float)":np.float32,"tensor(float16)":np.float16,"tensor(int64)":np.int64}
# Classify inputs by name; pair every state input with its output by name.
state={}  # input name -> array
pairs={}  # output name -> input name
for i in s.get_inputs():
    n=i.name
    if n.startswith("past_key_values."):
        state[n]=np.zeros([1,i.shape[1],0,i.shape[3]],TY[i.type]); pairs[n.replace("past_key_values.","present.")]=n
    elif n.startswith("past_conv."):
        state[n]=np.zeros([1,i.shape[1],i.shape[2]],TY[i.type]); pairs[n.replace("past_conv.","present_conv.")]=n
    elif n not in ("input_ids","attention_mask","position_ids","num_logits_to_keep"):
        raise SystemExit("unknown input "+n)
names=[o.name for o in s.get_outputs()]
assert all(o in names for o in pairs), "unpaired state"
out=[]; cur=ids; total=0
for step in range(32):
    total+=len(cur)
    feed={"input_ids":np.array([cur],np.int64),"attention_mask":np.ones([1,total],np.int64),"num_logits_to_keep":np.array(1,np.int64),**state}
    res=dict(zip(names,s.run(names,feed)))
    for o,i in pairs.items(): state[i]=res[o]
    tok=int(res["logits"][0,-1].argmax())
    if tok==7: break
    out.append(tok); cur=[tok]
print("logits shape",res["logits"].shape, "conv state", state["past_conv.0"].shape)
print("REPLY:",repr(decode(out)))
