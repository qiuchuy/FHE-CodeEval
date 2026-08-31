Example task: Element-wise square (`x * x`)

Implement this example in `fhe_kernel.py`.

### Input Specification

  5 random test inputs in [-1, 1]
  Accuracy metric: torch.allclose (atol=0.01, rtol=0.01)

  - x: torch.Tensor shape=[64], dtype=float32, range=[-1, 1]
    [ciphertext: encrypted client data]

### Reference Source

```python
import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return x * x
```

### Answer

```python
import openfhe
import torch


MULTIPLICATIVE_DEPTH = 1
SCALING_MOD_SIZE = 50
RING_DIM = 8192
N_SLOTS = 64
ROTATION_INDICES = []
OUTPUT_SHAPE = [64]


def make_context():
    parameters = openfhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(MULTIPLICATIVE_DEPTH)
    parameters.SetScalingModSize(SCALING_MOD_SIZE)
    parameters.SetScalingTechnique(openfhe.ScalingTechnique.FIXEDMANUAL)
    parameters.SetSecurityLevel(openfhe.HEStd_128_classic)
    parameters.SetRingDim(RING_DIM)
    parameters.SetBatchSize(N_SLOTS)

    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, ROTATION_INDICES)
    return cc, keys


def encrypt(cc, keys, x):
    values = x.detach().cpu().tolist()
    plaintext = cc.MakeCKKSPackedPlaintext(values)
    ciphertext = cc.Encrypt(keys.publicKey, plaintext)
    return {"x": ciphertext}


def fhe_kernel(cc, keys, enc_inputs):
    x_ct = enc_inputs["x"]
    return cc.EvalMult(x_ct, x_ct)


def decrypt(cc, keys, ct_out):
    plaintext = cc.Decrypt(keys.secretKey, ct_out)
    plaintext.SetLength(N_SLOTS)
    values = plaintext.GetRealPackedValue()
    return torch.tensor(values[:N_SLOTS], dtype=torch.float32).reshape(OUTPUT_SHAPE)
```
