import onnxruntime as ort, numpy as np
so=ort.SessionOptions(); so.log_severity_level=3
S='/mnt/sdd1/models/sherpa'
def tryrun(lab,s,feed):
    try:
        o=s.run(None,feed); print(lab,'OK',[x.shape for x in o][:3], o[0].ravel()[:4])
        return o
    except Exception as e: print(lab,'FAIL',str(e).split('\n')[0][:200])
# T5 punct int32
s=ort.InferenceSession(f'{S}/text/sherpa-onnx-online-punct-en-2024-08-06/model.onnx',so,providers=['CPUExecutionProvider'])
L=6; ids=np.arange(100,100+L)
tryrun('punct int64 tokens (node behaviour)',s,{'token_ids':ids[None].astype(np.int64),'valid_ids':np.array([[1]],np.int64),'label_lens':np.array([1],np.int64)})
tryrun('punct int32, node-shaped aux [1,1]',s,{'token_ids':ids[None].astype(np.int32),'valid_ids':np.array([[1]],np.int32),'label_lens':np.array([1],np.int32)})
tryrun('punct int32 correct aux',s,{'token_ids':ids[None].astype(np.int32),'valid_ids':np.ones((1,L),np.int32),'label_lens':np.array([L],np.int32)})
s=ort.InferenceSession(f'{S}/text/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx',so,providers=['CPUExecutionProvider'])
tryrun('ct-transformer int64',s,{'inputs':ids[None].astype(np.int64),'text_lengths':np.array([L],np.int64)})
tryrun('ct-transformer int32',s,{'inputs':ids[None].astype(np.int32),'text_lengths':np.array([L],np.int32)})
# T6 BERT
s=ort.InferenceSession('/mnt/sdd1/models/wild2/network_intrusion_detection__bert-network-packet-flow-header-payload__model.onnx',so,providers=['CPUExecutionProvider'])
ids=np.array([[101,2000,2001,2002,2003,102]],np.int64)
a=tryrun('bert mask=[1,1] fill round(0.667)=1 (node)',s,{'input_ids':ids,'attention_mask':np.array([[1]],np.int64)})
b=tryrun('bert mask=ones[1,L] (correct)',s,{'input_ids':ids,'attention_mask':np.ones_like(ids)})
tryrun('bert L=1, mask [1,1]',s,{'input_ids':ids[:,:1],'attention_mask':np.array([[1]],np.int64)})
tryrun('bert mask=[1,1] fill 0 (Param1<0.5)',s,{'input_ids':ids,'attention_mask':np.array([[0]],np.int64)})
if a is not None and b is not None: print('bert [1,1] vs ones max|diff|', np.abs(a[0]-b[0]).max())
