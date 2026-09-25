import onnx, sys, re
for p in sys.argv[1:]:
    m=onnx.load(p,load_external_data=False); g=m.graph
    init={i.name for i in g.initializer}
    ins=[i.name for i in g.input if i.name not in init]; outs=[o.name for o in g.output]
    # expected positional layout
    pk=[n for n in ins if n.startswith("past_key_values.")]
    L=len(pk)//2
    exp_in=["input_ids","attention_mask"]+(["position_ids"] if "position_ids" in ins else [])+[f"past_key_values.{l}.{kv}" for l in range(L) for kv in ("key","value")]
    exp_out=["logits"]+[f"present.{l}.{kv}" for l in range(L) for kv in ("key","value")]
    print(p.split('/')[-3], "in-ok" if ins==exp_in else f"IN-MISMATCH extra={[n for n in ins if n not in exp_in]}", "out-ok" if outs==exp_out else f"OUT-MISMATCH extra={[n for n in outs if n not in exp_out]}")
