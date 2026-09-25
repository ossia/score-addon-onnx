# Generates the fixtures of test_audio_overlap.cpp.
import os
import onnx
from onnx import TensorProto, helper

here = os.path.dirname(os.path.abspath(__file__))

def save(name, nodes, ins, outs):
    g = helper.make_graph(nodes, name, ins, outs)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(here, name + ".onnx"))

# A frame-based model that returns its 512-sample frame unchanged.
save("identity_512",
     [helper.make_node("Identity", ["audio"], ["out"])],
     [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, 512])],
     [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 512])])


# A model whose length is free: [1,1,N] -> the same.
save("identity_dyn",
     [helper.make_node("Identity", ["audio"], ["out"])],
     [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "N"])],
     [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, "N"])])

# Reference log-mel (HiFi-GAN settings, librosa's Slaney filterbank written
# out with numpy) of one streaming block of a fixed three-tone signal, for
# the MelFrontend test: float32 [80, 32].
import numpy as np

def hz_to_mel(f):
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    f = np.asarray(f, dtype=np.float64)
    return np.where(f < min_log_hz, f / f_sp,
                    min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep)

def mel_to_hz(m):
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    m = np.asarray(m, dtype=np.float64)
    return np.where(m < min_log_mel, m * f_sp,
                    min_log_hz * np.exp(logstep * (m - min_log_mel)))

def hz_to_mel_htk(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

def mel_to_hz_htk(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

def reference(sr, n_mels, fmin, fmax, htk, norm, floor, eps, name, n_fft=1024, hop=256, T=32):
    to_mel, to_hz = (hz_to_mel_htk, mel_to_hz_htk) if htk else (hz_to_mel, mel_to_hz)
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = to_hz(np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, len(fftfreqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    if norm:
        weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    n = T * hop + n_fft - hop
    t = np.arange(n)
    x = (0.3 * np.sin(2 * np.pi * 440 * t / sr)
         + 0.2 * np.sin(2 * np.pi * 3000 * t / sr + 0.5)
         + 0.05 * np.sin(2 * np.pi * 7000 * t / sr)).astype(np.float32).astype(np.float64)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    frames = np.stack([x[k * hop:k * hop + n_fft] * window for k in range(T)])
    spec = np.fft.rfft(frames, axis=1)
    mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + eps)
    logmel = np.log(np.maximum(weights @ mag.T, floor))
    logmel.astype(np.float32).tofile(os.path.join(here, name))

# HiFi-GAN: Slaney scale and norm, 0..8 kHz, sqrt(|X|^2 + 1e-9), floor 1e-5.
reference(22050, 80, 0.0, 8000.0, False, True, 1e-5, 1e-9, "mel_ref_80x32.f32")
# Vocos (torchaudio MelSpectrogram): HTK scale, no norm, 0..Nyquist, |X|,
# floor 1e-7.
reference(24000, 100, 0.0, 12000.0, True, False, 1e-7, 0.0, "mel_ref_vocos_100x32.f32")
