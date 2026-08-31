"""
Cleartext simulation backend for OpenFHE CKKS.

The FHE-CodeEval evaluator temporarily exposes this module as ``openfhe``
during simulator-backed checks. It replaces cryptographic operations with
NumPy equivalents while tracking ciphertext levels and SIMD slot widths.

Only CKKS scheme is supported. BFV/BGV/BinFHE are stubbed out minimally.
"""

from __future__ import annotations

import math
import copy
from enum import IntEnum
from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.polynomial import chebyshev

# ---------------------------------------------------------------------------
# Paterson-Stockmeyer depth table (matches OpenFHE ckksrns-utils.cpp)
# Maps (upper_bound_degree, depth).  For degree d, find the first entry
# whose upper_bound >= d; the corresponding depth is the multiplicative
# depth consumed by polynomial/Chebyshev evaluation of that degree.
# ---------------------------------------------------------------------------
_PS_DEPTH_TABLE: list[tuple[int, int]] = [
    (    0,  0),
    (    1,  1),
    (    2,  2),
    (    4,  3),
    (    5,  4),
    (   13,  5),
    (   27,  6),
    (   59,  7),
    (  119,  8),
    (  247,  9),
    (  495, 10),
    ( 1007, 11),
    ( 2031, 12),
    ( 4031, 13),
    ( 8127, 14),
    (16255, 15),
    (32639, 16),
    (65279, 17),
    (130815, 18),
    (261631, 19),
]

def _get_depth_by_degree(d: int) -> int:
    """Replicate OpenFHE's GetDepthByDegree lookup."""
    for upper, depth in _PS_DEPTH_TABLE:
        if d <= upper:
            return depth
    return max(1, math.ceil(math.log2(max(d, 1))) + 1)

# ---------------------------------------------------------------------------
# Enums  (values match the real OpenFHE C++ enum integers)
# ---------------------------------------------------------------------------

class SCHEME(IntEnum):
    INVALID_SCHEME = 0
    CKKSRNS_SCHEME = 1
    BFVRNS_SCHEME = 2
    BGVRNS_SCHEME = 3

class PKESchemeFeature(IntEnum):
    PKE = 1
    KEYSWITCH = 2
    PRE = 4
    LEVELEDSHE = 8
    ADVANCEDSHE = 16
    MULTIPARTY = 32
    FHE = 64
    SCHEMESWITCH = 128

class SecurityLevel(IntEnum):
    HEStd_128_classic = 0
    HEStd_128_quantum = 1
    HEStd_192_classic = 2
    HEStd_192_quantum = 3
    HEStd_256_classic = 4
    HEStd_256_quantum = 5
    HEStd_NotSet = 6

class ScalingTechnique(IntEnum):
    FIXEDMANUAL = 0
    FIXEDAUTO = 1
    FLEXIBLEAUTO = 2
    FLEXIBLEAUTOEXT = 3
    NORESCALE = 4
    INVALID_RS_TECHNIQUE = 5
    COMPOSITESCALINGAUTO = 6
    COMPOSITESCALINGMANUAL = 7

class SecretKeyDist(IntEnum):
    GAUSSIAN = 0
    UNIFORM_TERNARY = 1
    SPARSE_TERNARY = 2
    SPARSE_ENCAPSULATED = 3

class CKKSDataType(IntEnum):
    REAL = 0
    COMPLEX = 1

class Format(IntEnum):
    EVALUATION = 0
    COEFFICIENT = 1

class KeySwitchTechnique(IntEnum):
    INVALID_KS_TECH = 0
    BV = 1
    HYBRID = 2

class EncryptionTechnique(IntEnum):
    STANDARD = 0
    EXTENDED = 1

class MultiplicationTechnique(IntEnum):
    BEHZ = 0
    HPS = 1
    HPSPOVERQ = 2
    HPSPOVERQLEVELED = 3

class ProxyReEncryptionMode(IntEnum):
    NOT_SET = 0
    INDCPA = 1
    FIXED_NOISE_HRA = 2
    NOISE_FLOODING_HRA = 3

class MultipartyMode(IntEnum):
    INVALID_MULTIPARTY_MODE = 0
    FIXED_NOISE_MULTIPARTY = 1
    NOISE_FLOODING_MULTIPARTY = 2

class ExecutionMode(IntEnum):
    EXEC_EVALUATION = 0
    EXEC_NOISE_ESTIMATION = 1

class DecryptionNoiseMode(IntEnum):
    FIXED_NOISE_DECRYPT = 0
    NOISE_FLOODING_DECRYPT = 1

class CompressionLevel(IntEnum):
    COMPACT = 0
    SLACK = 1

class _SerType:
    pass

SERJSON = _SerType()
SERBINARY = _SerType()

# ---------------------------------------------------------------------------
# Module-level aliases (match the real openfhe module namespace)
# ---------------------------------------------------------------------------

INVALID_SCHEME = SCHEME.INVALID_SCHEME
CKKSRNS_SCHEME = SCHEME.CKKSRNS_SCHEME
BFVRNS_SCHEME = SCHEME.BFVRNS_SCHEME
BGVRNS_SCHEME = SCHEME.BGVRNS_SCHEME

PKE = PKESchemeFeature.PKE
KEYSWITCH = PKESchemeFeature.KEYSWITCH
PRE = PKESchemeFeature.PRE
LEVELEDSHE = PKESchemeFeature.LEVELEDSHE
ADVANCEDSHE = PKESchemeFeature.ADVANCEDSHE
MULTIPARTY = PKESchemeFeature.MULTIPARTY
FHE = PKESchemeFeature.FHE
SCHEMESWITCH = PKESchemeFeature.SCHEMESWITCH

HEStd_128_classic = SecurityLevel.HEStd_128_classic
HEStd_128_quantum = SecurityLevel.HEStd_128_quantum
HEStd_192_classic = SecurityLevel.HEStd_192_classic
HEStd_192_quantum = SecurityLevel.HEStd_192_quantum
HEStd_256_classic = SecurityLevel.HEStd_256_classic
HEStd_256_quantum = SecurityLevel.HEStd_256_quantum
HEStd_NotSet = SecurityLevel.HEStd_NotSet

FIXEDMANUAL = ScalingTechnique.FIXEDMANUAL
FIXEDAUTO = ScalingTechnique.FIXEDAUTO
FLEXIBLEAUTO = ScalingTechnique.FLEXIBLEAUTO
FLEXIBLEAUTOEXT = ScalingTechnique.FLEXIBLEAUTOEXT
NORESCALE = ScalingTechnique.NORESCALE
INVALID_RS_TECHNIQUE = ScalingTechnique.INVALID_RS_TECHNIQUE
COMPOSITESCALINGAUTO = ScalingTechnique.COMPOSITESCALINGAUTO
COMPOSITESCALINGMANUAL = ScalingTechnique.COMPOSITESCALINGMANUAL

GAUSSIAN = SecretKeyDist.GAUSSIAN
UNIFORM_TERNARY = SecretKeyDist.UNIFORM_TERNARY
SPARSE_TERNARY = SecretKeyDist.SPARSE_TERNARY
SPARSE_ENCAPSULATED = SecretKeyDist.SPARSE_ENCAPSULATED

REAL = CKKSDataType.REAL
COMPLEX = CKKSDataType.COMPLEX

EVALUATION = Format.EVALUATION
COEFFICIENT = Format.COEFFICIENT

INVALID_KS_TECH = KeySwitchTechnique.INVALID_KS_TECH
BV = KeySwitchTechnique.BV
HYBRID = KeySwitchTechnique.HYBRID

STANDARD = EncryptionTechnique.STANDARD
EXTENDED = EncryptionTechnique.EXTENDED

BEHZ = MultiplicationTechnique.BEHZ
HPS = MultiplicationTechnique.HPS
HPSPOVERQ = MultiplicationTechnique.HPSPOVERQ
HPSPOVERQLEVELED = MultiplicationTechnique.HPSPOVERQLEVELED

NOT_SET = ProxyReEncryptionMode.NOT_SET
INDCPA = ProxyReEncryptionMode.INDCPA
FIXED_NOISE_HRA = ProxyReEncryptionMode.FIXED_NOISE_HRA
NOISE_FLOODING_HRA = ProxyReEncryptionMode.NOISE_FLOODING_HRA

INVALID_MULTIPARTY_MODE = MultipartyMode.INVALID_MULTIPARTY_MODE
FIXED_NOISE_MULTIPARTY = MultipartyMode.FIXED_NOISE_MULTIPARTY
NOISE_FLOODING_MULTIPARTY = MultipartyMode.NOISE_FLOODING_MULTIPARTY

EXEC_EVALUATION = ExecutionMode.EXEC_EVALUATION
EXEC_NOISE_ESTIMATION = ExecutionMode.EXEC_NOISE_ESTIMATION

FIXED_NOISE_DECRYPT = DecryptionNoiseMode.FIXED_NOISE_DECRYPT
NOISE_FLOODING_DECRYPT = DecryptionNoiseMode.NOISE_FLOODING_DECRYPT

COMPACT = CompressionLevel.COMPACT
SLACK = CompressionLevel.SLACK

JSON = SERJSON
BINARY = SERBINARY

# BinFHE stubs (minimal, enough so `from openfhe import *` doesn't break)

class BINFHE_PARAMSET(IntEnum):
    TOY = 0; STD128 = 1; STD128_AP = 2; STD128_3 = 3; STD128_4 = 4
    STD128Q = 5; STD128Q_3 = 6; STD128Q_4 = 7
    STD192 = 8; STD192Q = 9; STD192Q_3 = 10; STD192Q_4 = 11
    STD256 = 12; STD256Q = 13; STD256Q_3 = 14; STD256Q_4 = 15
    STD128_LMKCDEY = 16; STD128Q_LMKCDEY = 17
    STD128_3_LMKCDEY = 18; STD128Q_3_LMKCDEY = 19
    STD128_4_LMKCDEY = 20; STD128Q_4_LMKCDEY = 21
    SIGNED_MOD_TEST = 22

TOY = BINFHE_PARAMSET.TOY
STD128 = BINFHE_PARAMSET.STD128
STD128_AP = BINFHE_PARAMSET.STD128_AP
STD128_3 = BINFHE_PARAMSET.STD128_3
STD128_4 = BINFHE_PARAMSET.STD128_4
STD128Q = BINFHE_PARAMSET.STD128Q
STD128Q_3 = BINFHE_PARAMSET.STD128Q_3
STD128Q_4 = BINFHE_PARAMSET.STD128Q_4
STD128_LMKCDEY = BINFHE_PARAMSET.STD128_LMKCDEY
STD128Q_LMKCDEY = BINFHE_PARAMSET.STD128Q_LMKCDEY
STD128_3_LMKCDEY = BINFHE_PARAMSET.STD128_3_LMKCDEY
STD128Q_3_LMKCDEY = BINFHE_PARAMSET.STD128Q_3_LMKCDEY
STD128_4_LMKCDEY = BINFHE_PARAMSET.STD128_4_LMKCDEY
STD128Q_4_LMKCDEY = BINFHE_PARAMSET.STD128Q_4_LMKCDEY
STD192 = BINFHE_PARAMSET.STD192
STD192Q = BINFHE_PARAMSET.STD192Q
STD192Q_3 = BINFHE_PARAMSET.STD192Q_3
STD192Q_4 = BINFHE_PARAMSET.STD192Q_4
STD256 = BINFHE_PARAMSET.STD256
STD256Q = BINFHE_PARAMSET.STD256Q
STD256Q_3 = BINFHE_PARAMSET.STD256Q_3
STD256Q_4 = BINFHE_PARAMSET.STD256Q_4
SIGNED_MOD_TEST = BINFHE_PARAMSET.SIGNED_MOD_TEST

class BINFHE_METHOD(IntEnum):
    INVALID_METHOD = 0; AP = 1; GINX = 2; LMKCDEY = 3

AP = BINFHE_METHOD.AP
GINX = BINFHE_METHOD.GINX
LMKCDEY = BINFHE_METHOD.LMKCDEY
INVALID_METHOD = BINFHE_METHOD.INVALID_METHOD

class BINFHE_OUTPUT(IntEnum):
    INVALID_OUTPUT = 0; FRESH = 1; BOOTSTRAPPED = 2

FRESH = BINFHE_OUTPUT.FRESH
BOOTSTRAPPED = BINFHE_OUTPUT.BOOTSTRAPPED
INVALID_OUTPUT = BINFHE_OUTPUT.INVALID_OUTPUT

class BINGATE(IntEnum):
    OR = 0; AND = 1; NOR = 2; NAND = 3; XOR = 4; XNOR = 5
    XOR_FAST = 6; XNOR_FAST = 7

OR = BINGATE.OR
AND = BINGATE.AND
NOR = BINGATE.NOR
NAND = BINGATE.NAND
XOR = BINGATE.XOR
XNOR = BINGATE.XNOR
XOR_FAST = BINGATE.XOR_FAST
XNOR_FAST = BINGATE.XNOR_FAST

class KEYGEN_MODE(IntEnum):
    SYM_ENCRYPT = 0; PUB_ENCRYPT = 1

SYM_ENCRYPT = KEYGEN_MODE.SYM_ENCRYPT
PUB_ENCRYPT = KEYGEN_MODE.PUB_ENCRYPT


def get_native_int() -> int:
    return 64

# ---------------------------------------------------------------------------
# Stub key / param types
# ---------------------------------------------------------------------------

class DCRTPoly:
    pass

class ParmType:
    pass

class PublicKey:
    def __init__(self):
        self._tag = ""
    def GetKeyTag(self) -> str:
        return self._tag
    def SetKeyTag(self, tag: str):
        self._tag = tag

class PrivateKey:
    def __init__(self):
        self._tag = ""
        self._cc = None
    def GetCryptoContext(self):
        return self._cc
    def GetKeyTag(self) -> str:
        return self._tag
    def SetKeyTag(self, tag: str):
        self._tag = tag

class KeyPair:
    def __init__(self):
        self.publicKey = PublicKey()
        self.secretKey = PrivateKey()
    def good(self) -> bool:
        return True

class EvalKey:
    def __init__(self):
        self._tag = ""
    def GetKeyTag(self) -> str:
        return self._tag
    def SetKeyTag(self, tag: str):
        self._tag = tag

class EvalKeyMap(dict):
    pass

class SchSwchParams:
    def __init__(self):
        self._attrs: dict[str, Any] = {}
    def __getattr__(self, name: str):
        if name.startswith("Get"):
            key = name[3:]
            return lambda: self._attrs.get(key, 0)
        if name.startswith("Set"):
            key = name[3:]
            return lambda v: self._attrs.__setitem__(key, v)
        raise AttributeError(name)
    def __str__(self):
        return f"SchSwchParams({self._attrs})"


# Minimal BinFHE stubs
class LWEPrivateKey:
    def __init__(self): pass
    def GetLength(self): return 0
class LWECiphertext:
    def __init__(self): pass
    def GetLength(self): return 0
    def GetModulus(self): return 0
class BinFHEContext:
    pass

# ---------------------------------------------------------------------------
# CCParamsCKKSRNS
# ---------------------------------------------------------------------------

class CCParamsCKKSRNS:
    def __init__(self):
        self._multiplicative_depth: int = 1
        self._scaling_mod_size: int = 50
        self._batch_size: int = 0
        self._first_mod_size: int = 60
        self._ring_dim: int = 0
        self._security_level = SecurityLevel.HEStd_128_classic
        self._scaling_technique = ScalingTechnique.FLEXIBLEAUTO
        self._key_switch_technique = KeySwitchTechnique.HYBRID
        self._secret_key_dist = SecretKeyDist.UNIFORM_TERNARY
        self._num_large_digits: int = 0
        self._plaintext_modulus: int = 0
        self._digit_size: int = 0
        self._standard_deviation: float = 3.2
        self._eval_add_count: int = 0
        self._key_switch_count: int = 0
        self._encryption_technique = EncryptionTechnique.STANDARD
        self._multiplication_technique = MultiplicationTechnique.HPS
        self._max_relin_sk_deg: int = 2
        self._pre_mode = ProxyReEncryptionMode.NOT_SET
        self._multiparty_mode = MultipartyMode.INVALID_MULTIPARTY_MODE
        self._execution_mode = ExecutionMode.EXEC_EVALUATION
        self._decryption_noise_mode = DecryptionNoiseMode.FIXED_NOISE_DECRYPT
        self._noise_estimate: float = 0.0
        self._desired_precision: float = 25.0
        self._statistical_security: int = 30
        self._num_adversarial_queries: int = 1
        self._pre_num_hops: int = 1
        self._interactive_boot_compression_level = CompressionLevel.SLACK
        self._ckks_data_type = CKKSDataType.REAL
        self._composite_degree: int = 0
        self._register_word_size: int = 0

    # Getters
    def GetMultiplicativeDepth(self) -> int: return self._multiplicative_depth
    def GetScalingModSize(self) -> int: return self._scaling_mod_size
    def GetBatchSize(self) -> int: return self._batch_size
    def GetFirstModSize(self) -> int: return self._first_mod_size
    def GetRingDim(self) -> int: return self._ring_dim
    def GetSecurityLevel(self): return self._security_level
    def GetScalingTechnique(self): return self._scaling_technique
    def GetKeySwitchTechnique(self): return self._key_switch_technique
    def GetSecretKeyDist(self): return self._secret_key_dist
    def GetNumLargeDigits(self) -> int: return self._num_large_digits
    def GetPlaintextModulus(self) -> int: return self._plaintext_modulus
    def GetDigitSize(self) -> int: return self._digit_size
    def GetStandardDeviation(self) -> float: return self._standard_deviation
    def GetEvalAddCount(self) -> int: return self._eval_add_count
    def GetKeySwitchCount(self) -> int: return self._key_switch_count
    def GetEncryptionTechnique(self): return self._encryption_technique
    def GetMultiplicationTechnique(self): return self._multiplication_technique
    def GetMaxRelinSkDeg(self) -> int: return self._max_relin_sk_deg
    def GetPREMode(self): return self._pre_mode
    def GetMultipartyMode(self): return self._multiparty_mode
    def GetExecutionMode(self): return self._execution_mode
    def GetDecryptionNoiseMode(self): return self._decryption_noise_mode
    def GetNoiseEstimate(self) -> float: return self._noise_estimate
    def GetDesiredPrecision(self) -> float: return self._desired_precision
    def GetStatisticalSecurity(self) -> int: return self._statistical_security
    def GetNumAdversarialQueries(self) -> int: return self._num_adversarial_queries
    def GetPRENumHops(self) -> int: return self._pre_num_hops
    def GetInteractiveBootCompressionLevel(self): return self._interactive_boot_compression_level
    def GetCKKSDataType(self): return self._ckks_data_type
    def GetScheme(self): return SCHEME.CKKSRNS_SCHEME
    def GetCompositeDegree(self) -> int: return self._composite_degree
    def GetRegisterWordSize(self) -> int: return self._register_word_size

    # Setters
    def SetMultiplicativeDepth(self, v: int): self._multiplicative_depth = v
    def SetScalingModSize(self, v: int): self._scaling_mod_size = v
    def SetBatchSize(self, v: int): self._batch_size = v
    def SetFirstModSize(self, v: int): self._first_mod_size = v
    def SetRingDim(self, v: int): self._ring_dim = v
    def SetSecurityLevel(self, v): self._security_level = v
    def SetScalingTechnique(self, v): self._scaling_technique = v
    def SetKeySwitchTechnique(self, v): self._key_switch_technique = v
    def SetSecretKeyDist(self, v): self._secret_key_dist = v
    def SetNumLargeDigits(self, v: int): self._num_large_digits = v
    def SetPlaintextModulus(self, v: int): self._plaintext_modulus = v
    def SetDigitSize(self, v: int): self._digit_size = v
    def SetStandardDeviation(self, v: float): self._standard_deviation = v
    def SetEvalAddCount(self, v: int): self._eval_add_count = v
    def SetKeySwitchCount(self, v: int): self._key_switch_count = v
    def SetEncryptionTechnique(self, v): self._encryption_technique = v
    def SetMultiplicationTechnique(self, v): self._multiplication_technique = v
    def SetMaxRelinSkDeg(self, v: int): self._max_relin_sk_deg = v
    def SetPREMode(self, v): self._pre_mode = v
    def SetMultipartyMode(self, v): self._multiparty_mode = v
    def SetExecutionMode(self, v): self._execution_mode = v
    def SetDecryptionNoiseMode(self, v): self._decryption_noise_mode = v
    def SetNoiseEstimate(self, v: float): self._noise_estimate = v
    def SetDesiredPrecision(self, v: float): self._desired_precision = v
    def SetStatisticalSecurity(self, v: int): self._statistical_security = v
    def SetNumAdversarialQueries(self, v: int): self._num_adversarial_queries = v
    def SetThresholdNumOfParties(self, v: int): pass
    def SetPRENumHops(self, v: int): self._pre_num_hops = v
    def SetInteractiveBootCompressionLevel(self, v): self._interactive_boot_compression_level = v
    def SetCKKSDataType(self, v): self._ckks_data_type = v
    def SetCompositeDegree(self, v: int): self._composite_degree = v
    def SetRegisterWordSize(self, v: int): self._register_word_size = v

    def __str__(self):
        return (f"CCParamsCKKSRNS(depth={self._multiplicative_depth}, "
                f"scalingModSize={self._scaling_mod_size}, "
                f"batchSize={self._batch_size})")


# Stubs for BFV/BGV params (not functional, just enough to not crash imports)
class CCParamsBFVRNS(CCParamsCKKSRNS):
    def GetScheme(self): return SCHEME.BFVRNS_SCHEME
class CCParamsBGVRNS(CCParamsCKKSRNS):
    def GetScheme(self): return SCHEME.BGVRNS_SCHEME


# ---------------------------------------------------------------------------
# Plaintext
# ---------------------------------------------------------------------------

class Plaintext:
    def __init__(self, data: np.ndarray, slots: int = 0, level: int = 0,
                 noise_scale_deg: int = 1, scaling_factor: float = 1.0,
                 ckks_data_type: CKKSDataType = CKKSDataType.REAL):
        self._data = np.array(data, dtype=np.float64)
        self._slots = slots if slots > 0 else len(self._data)
        self._level = level
        self._noise_scale_deg = noise_scale_deg
        self._scaling_factor = scaling_factor
        self._length: Optional[int] = None
        self._ckks_data_type = ckks_data_type
        self._scheme_id = SCHEME.CKKSRNS_SCHEME
        self._format = Format.EVALUATION
        self._string_value = ""

    def GetScalingFactor(self) -> float:
        return self._scaling_factor
    def SetScalingFactor(self, v: float):
        self._scaling_factor = v
    def GetSchemeID(self):
        return self._scheme_id
    def GetLength(self) -> int:
        if self._length is not None:
            return self._length
        return len(self._data)
    def SetLength(self, n: int):
        self._length = n
    def IsEncoded(self) -> bool:
        return True
    def GetLogPrecision(self) -> int:
        return 50
    def Encode(self):
        pass
    def Decode(self, *args):
        pass
    def LowBound(self) -> float:
        return float(np.min(self._data))
    def HighBound(self) -> float:
        return float(np.max(self._data))
    def SetFormat(self, fmt):
        self._format = fmt

    def GetCoefPackedValue(self) -> list:
        return self._data.tolist()

    def GetPackedValue(self) -> list:
        return self._data.astype(np.int64).tolist()

    def GetCKKSPackedValue(self) -> list:
        n = self._length if self._length is not None else len(self._data)
        return [complex(v, 0) for v in self._data[:n]]

    def GetRealPackedValue(self) -> list:
        n = self._length if self._length is not None else len(self._data)
        return self._data[:n].tolist()

    def GetLevel(self) -> int:
        return self._level
    def SetLevel(self, v: int):
        self._level = v
    def GetNoiseScaleDeg(self) -> int:
        return self._noise_scale_deg
    def SetNoiseScaleDeg(self, v: int):
        self._noise_scale_deg = v
    def GetSlots(self) -> int:
        return self._slots
    def SetSlots(self, v: int):
        self._slots = v
    def GetLogError(self) -> float:
        return 0.0
    def GetStringValue(self) -> str:
        return self._string_value
    def SetStringValue(self, v: str):
        self._string_value = v
    def SetIntVectorValue(self, v: list):
        self._data = np.array(v, dtype=np.float64)
    def GetFormattedValues(self, *args) -> str:
        return str(self._data)

    def __repr__(self):
        return f"Plaintext(level={self._level}, slots={self._slots}, len={len(self._data)})"
    def __str__(self):
        return self.__repr__()


# ---------------------------------------------------------------------------
# Ciphertext
# ---------------------------------------------------------------------------

class Ciphertext:
    def __init__(self, data: Optional[np.ndarray] = None, level: int = 0,
                 slots: int = 0, scaling_factor: float = 1.0,
                 noise_scale_deg: int = 1,
                 crypto_context: Optional["CryptoContext"] = None):
        self._data = np.array(data, dtype=np.float64) if data is not None else np.zeros(0, dtype=np.float64)
        self._level = level
        self._slots = slots
        self._scaling_factor = scaling_factor
        self._noise_scale_deg = noise_scale_deg
        self._cc = crypto_context
        self._key_tag = ""
        self._encoding_type = SCHEME.CKKSRNS_SCHEME

    def GetLevel(self) -> int:
        return self._level
    def SetLevel(self, v: int):
        self._level = v
    def GetSlots(self) -> int:
        return self._slots
    def SetSlots(self, v: int):
        self._slots = v
    def GetScalingFactor(self) -> float:
        return self._scaling_factor
    def SetScalingFactor(self, v: float):
        self._scaling_factor = v
    def GetNoiseScaleDeg(self) -> int:
        return self._noise_scale_deg
    def SetNoiseScaleDeg(self, v: int):
        self._noise_scale_deg = v
    def GetCryptoContext(self):
        return self._cc
    def GetKeyTag(self) -> str:
        return self._key_tag
    def GetEncodingType(self):
        return self._encoding_type

    def Clone(self) -> "Ciphertext":
        ct = Ciphertext(
            data=self._data.copy(),
            level=self._level,
            slots=self._slots,
            scaling_factor=self._scaling_factor,
            noise_scale_deg=self._noise_scale_deg,
            crypto_context=self._cc,
        )
        ct._key_tag = self._key_tag
        return ct

    def RemoveElement(self, idx: int):
        pass

    def GetElements(self):
        return [self._data]
    def GetElementsMutable(self):
        return [self._data]
    def SetElements(self, elems):
        if elems:
            self._data = np.array(elems[0], dtype=np.float64)
    def SetElementsMove(self, elems):
        self.SetElements(elems)

    def __add__(self, other):
        if self._cc is not None:
            return self._cc.EvalAdd(self, other)
        if isinstance(other, Ciphertext):
            return Ciphertext(data=self._data + other._data,
                              level=max(self._level, other._level),
                              slots=self._slots, crypto_context=self._cc)
        return NotImplemented

    def __getattr__(self, name: str):
        def _unimplemented_stub(*args, **kwargs):
            return None
        return _unimplemented_stub

    def __repr__(self):
        return f"Ciphertext(level={self._level}, slots={self._slots})"


# ---------------------------------------------------------------------------
# CryptoContext
# ---------------------------------------------------------------------------

class CryptoContext:
    def __init__(self, params: CCParamsCKKSRNS):
        self._params = params
        self._mult_depth = params.GetMultiplicativeDepth()
        self._batch_size = params.GetBatchSize()
        self._ring_dim = params.GetRingDim()
        self._scaling_mod_size = params.GetScalingModSize()
        self._first_mod_size = params.GetFirstModSize()
        self._scaling_technique = params.GetScalingTechnique()
        self._secret_key_dist = params.GetSecretKeyDist()
        self._ckks_data_type = params.GetCKKSDataType()
        self._enabled_features: set = set()
        self._key_gen_level: int = 0
        self._bootstrap_depth: int = 0
        self._ckks_boot_correction_factor: float = 0.0

        if self._ring_dim == 0:
            self._ring_dim = 2 ** max(12, math.ceil(math.log2(
                max(16, self._batch_size * 2 if self._batch_size > 0 else 4096)
            )))
        if self._batch_size <= 0:
            self._batch_size = self._ring_dim // 2

    # --- Context queries ---
    def GetKeyGenLevel(self) -> int:
        return self._key_gen_level
    def SetKeyGenLevel(self, v: int):
        self._key_gen_level = v
    def get_ptr(self):
        print(f"SimCryptoContext @ {id(self)}")
    def GetRingDimension(self) -> int:
        return self._ring_dim
    def GetPlaintextModulus(self) -> int:
        return 0
    def GetBatchSize(self) -> int:
        return self._batch_size
    def GetModulus(self):
        return 1 << (self._scaling_mod_size * (self._mult_depth + 1))
    def GetModulusCKKS(self) -> float:
        return float(1 << (self._scaling_mod_size * (self._mult_depth + 1)))
    def GetScalingFactorReal(self, level: int = 0) -> float:
        return float(1 << self._scaling_mod_size)
    def GetScalingTechnique(self):
        return self._scaling_technique
    def GetDigitSize(self) -> int:
        return self._params.GetDigitSize()
    def GetCyclotomicOrder(self) -> int:
        return self._ring_dim * 2
    def GetCKKSDataType(self):
        return self._ckks_data_type
    def GetCKKSBootCorrectionFactor(self) -> float:
        return self._ckks_boot_correction_factor
    def SetCKKSBootCorrectionFactor(self, v: float):
        self._ckks_boot_correction_factor = v

    def GetNoiseEstimate(self) -> float: return self._params.GetNoiseEstimate()
    def SetNoiseEstimate(self, v: float): self._params.SetNoiseEstimate(v)
    def GetMultiplicativeDepth(self) -> int: return self._mult_depth
    def SetMultiplicativeDepth(self, v: int): self._mult_depth = v
    def GetEvalAddCount(self) -> int: return self._params.GetEvalAddCount()
    def SetEvalAddCount(self, v: int): self._params.SetEvalAddCount(v)
    def GetKeySwitchCount(self) -> int: return self._params.GetKeySwitchCount()
    def SetKeySwitchCount(self, v: int): self._params.SetKeySwitchCount(v)
    def GetPRENumHops(self) -> int: return self._params.GetPRENumHops()
    def SetPRENumHops(self, v: int): self._params.SetPRENumHops(v)
    def GetRegisterWordSize(self) -> int: return self._params.GetRegisterWordSize()
    def GetCompositeDegree(self) -> int: return self._params.GetCompositeDegree()
    def GetKeySwitchTechnique(self): return self._params.GetKeySwitchTechnique()

    # --- Enable / keygen ---
    def Enable(self, feature):
        self._enabled_features.add(int(feature))

    def _verify_pke(self, op: str = ""):
        if int(PKESchemeFeature.PKE) not in self._enabled_features:
            raise RuntimeError(f"{op} operation has not been enabled. "
                               "Enable(PKE) must be called to enable it.")

    def _verify_leveledshe(self, op: str = ""):
        if int(PKESchemeFeature.LEVELEDSHE) not in self._enabled_features:
            raise RuntimeError(f"{op} operation has not been enabled. "
                               "Enable(LEVELEDSHE) must be called to enable it.")

    def _verify_advancedshe(self, op: str = ""):
        if int(PKESchemeFeature.ADVANCEDSHE) not in self._enabled_features:
            raise RuntimeError(f"{op} operation has not been enabled. "
                               "Enable(ADVANCEDSHE) must be called to enable it.")

    def _verify_fhe(self, op: str = ""):
        if int(PKESchemeFeature.FHE) not in self._enabled_features:
            raise RuntimeError(f"{op} operation has not been enabled. "
                               "Enable(FHE) must be called to enable it.")

    def KeyGen(self) -> KeyPair:
        self._verify_pke("KeyGen")
        kp = KeyPair()
        kp.secretKey._cc = self
        return kp

    def EvalMultKeyGen(self, secret_key): pass
    def EvalMultKeysGen(self, secret_key): pass
    def EvalRotateKeyGen(self, secret_key, indices=None): pass
    def EvalAtIndexKeyGen(self, secret_key, indices=None): pass
    def EvalSumKeyGen(self, secret_key, public_key=None): pass
    def EvalSumRowsKeyGen(self, secret_key, public_key=None, row_size=None): pass
    def EvalSumColsKeyGen(self, secret_key, public_key=None): pass
    def EvalAutomorphismKeyGen(self, secret_key, indices=None): pass
    def KeySwitchGen(self, old_key, new_key): return EvalKey()

    # --- Encoding ---
    def MakeStringPlaintext(self, s: str) -> Plaintext:
        data = np.array([float(ord(c)) for c in s], dtype=np.float64)
        return Plaintext(data, slots=len(data))

    def MakePackedPlaintext(self, values: list, noiseScaleDeg: int = 1, level: int = 0) -> Plaintext:
        data = np.array(values, dtype=np.float64)
        return Plaintext(data, slots=len(data), level=level, noise_scale_deg=noiseScaleDeg)

    def MakeCoefPackedPlaintext(self, values: list, noiseScaleDeg: int = 1, level: int = 0) -> Plaintext:
        data = np.array(values, dtype=np.float64)
        return Plaintext(data, slots=len(data), level=level, noise_scale_deg=noiseScaleDeg)

    def MakeCKKSPackedPlaintext(self, values, noiseScaleDeg: int = 1, level: int = 0,
                                 params=None, slots: int = 0) -> Plaintext:
        if hasattr(values, '__len__'):
            raw = [float(v.real) if isinstance(v, complex) else float(v) for v in values]
        else:
            raw = [float(values)]

        n = slots if slots > 0 else self._batch_size
        if len(raw) < n:
            raw = raw + [0.0] * (n - len(raw))
        data = np.array(raw[:n], dtype=np.float64)
        return Plaintext(data, slots=n, level=level, noise_scale_deg=noiseScaleDeg,
                         ckks_data_type=self._ckks_data_type)

    # --- Encrypt / Decrypt ---
    def Encrypt(self, key, plaintext: Plaintext) -> Ciphertext:
        self._verify_pke("Encrypt")
        return Ciphertext(
            data=plaintext._data.copy(),
            level=plaintext._level,
            slots=plaintext._slots,
            scaling_factor=plaintext._scaling_factor,
            noise_scale_deg=plaintext._noise_scale_deg,
            crypto_context=self,
        )

    def Decrypt(self, key_or_ct, ct_or_key) -> Plaintext:
        self._verify_pke("Decrypt")
        if isinstance(key_or_ct, Ciphertext):
            ct = key_or_ct
        elif isinstance(ct_or_key, Ciphertext):
            ct = ct_or_key
        else:
            raise TypeError("Decrypt expects a Ciphertext argument")
        # Mirror rns-pke.cpp: sizeQl = (mult_depth + 1) - level; fail if sizeQl == 0.
        sizeQl = (self._mult_depth + 1) - ct._level
        if sizeQl <= 0:
            raise RuntimeError(
                f"Decryption failure: No towers left; consider increasing the depth. "
                f"(ciphertext level={ct._level}, multiplicative_depth={self._mult_depth})"
            )
        return Plaintext(ct._data.copy(), slots=ct._slots, level=ct._level,
                         noise_scale_deg=ct._noise_scale_deg,
                         scaling_factor=ct._scaling_factor)

    # --- Level management helpers ---
    def _check_level(self, new_level: int, op_name: str):
        if new_level > self._mult_depth:
            raise RuntimeError(
                f"[CKKS Simulation] {op_name}: result level {new_level} "
                f"exceeds multiplicative depth {self._mult_depth}"
            )

    def _coerce_operand(self, x) -> tuple[np.ndarray, int, bool]:
        """Return (data, level, is_ciphertext) for an operand."""
        if isinstance(x, Ciphertext):
            return x._data, x._level, True
        if isinstance(x, Plaintext):
            return x._data, x._level, False
        return np.full(self._batch_size, float(x), dtype=np.float64), 0, False

    def _broadcast(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pad shorter array to match the longer one."""
        if len(a) == len(b):
            return a, b
        n = max(len(a), len(b))
        if len(a) < n:
            a = np.pad(a, (0, n - len(a)))
        if len(b) < n:
            b = np.pad(b, (0, n - len(b)))
        return a, b

    def _make_result_ct(self, data: np.ndarray, level: int, slots: int) -> Ciphertext:
        return Ciphertext(data=data, level=level, slots=slots, crypto_context=self)

    # --- Arithmetic ---
    def EvalAdd(self, a, b) -> Ciphertext:
        self._verify_leveledshe("EvalAdd")
        da, la, _ = self._coerce_operand(a)
        db, lb, _ = self._coerce_operand(b)
        da, db = self._broadcast(da, db)
        new_level = max(la, lb)
        slots = a._slots if isinstance(a, Ciphertext) else (b._slots if isinstance(b, Ciphertext) else len(da))
        return self._make_result_ct(da + db, new_level, slots)

    def EvalAddInPlace(self, a: Ciphertext, b) -> Ciphertext:
        self._verify_leveledshe("EvalAddInPlace")
        db, lb, _ = self._coerce_operand(b)
        da, db = self._broadcast(a._data, db)
        a._data = da + db
        a._level = max(a._level, lb)
        return a

    def EvalAddMutable(self, a, b) -> Ciphertext:
        return self.EvalAdd(a, b)
    def EvalAddMutableInPlace(self, a, b) -> Ciphertext:
        return self.EvalAddInPlace(a, b)

    def EvalSub(self, a, b) -> Ciphertext:
        self._verify_leveledshe("EvalSub")
        da, la, _ = self._coerce_operand(a)
        db, lb, _ = self._coerce_operand(b)
        da, db = self._broadcast(da, db)
        new_level = max(la, lb)
        slots = a._slots if isinstance(a, Ciphertext) else (b._slots if isinstance(b, Ciphertext) else len(da))
        return self._make_result_ct(da - db, new_level, slots)

    def EvalSubInPlace(self, a: Ciphertext, b) -> Ciphertext:
        self._verify_leveledshe("EvalSubInPlace")
        db, lb, _ = self._coerce_operand(b)
        da, db = self._broadcast(a._data, db)
        a._data = da - db
        a._level = max(a._level, lb)
        return a

    def EvalSubMutable(self, a, b) -> Ciphertext:
        return self.EvalSub(a, b)
    def EvalSubMutableInPlace(self, a, b) -> Ciphertext:
        return self.EvalSubInPlace(a, b)

    def EvalMult(self, a, b) -> Ciphertext:
        self._verify_leveledshe("EvalMult")
        da, la, a_is_ct = self._coerce_operand(a)
        db, lb, b_is_ct = self._coerce_operand(b)
        da, db = self._broadcast(da, db)
        # FIXEDMANUAL: multiplication never consumes a level; only Rescale/ModReduce does
        new_level = max(la, lb)
        slots = a._slots if isinstance(a, Ciphertext) else (b._slots if isinstance(b, Ciphertext) else len(da))
        return self._make_result_ct(da * db, new_level, slots)

    def EvalMultInPlace(self, a: Ciphertext, b) -> Ciphertext:
        self._verify_leveledshe("EvalMultInPlace")
        db, lb, b_is_ct = self._coerce_operand(b)
        da, db = self._broadcast(a._data, db)
        # FIXEDMANUAL: level not consumed here; only Rescale/ModReduce does
        a._data = da * db
        a._level = max(a._level, lb)
        return a

    def EvalMultMutable(self, a, b) -> Ciphertext:
        return self.EvalMult(a, b)
    def EvalMultMutableInPlace(self, a, b) -> Ciphertext:
        return self.EvalMultInPlace(a, b)

    def EvalMultNoRelin(self, a, b) -> Ciphertext:
        return self.EvalMult(a, b)

    def EvalMultAndRelinearize(self, a, b) -> Ciphertext:
        return self.EvalMult(a, b)

    def Relinearize(self, ct) -> Ciphertext:
        return ct.Clone()
    def RelinearizeInPlace(self, ct) -> Ciphertext:
        return ct

    def EvalSquare(self, ct: Ciphertext) -> Ciphertext:
        self._verify_leveledshe("EvalSquare")
        # FIXEDMANUAL: squaring is ct×ct; level consumed only when Rescale is called
        return self._make_result_ct(ct._data ** 2, ct._level, ct._slots)

    def EvalSquareMutable(self, ct: Ciphertext) -> Ciphertext:
        return self.EvalSquare(ct)

    def EvalSquareInPlace(self, ct: Ciphertext) -> Ciphertext:
        self._verify_leveledshe("EvalSquareInPlace")
        ct._data = ct._data ** 2
        return ct

    def EvalNegate(self, ct: Ciphertext) -> Ciphertext:
        self._verify_leveledshe("EvalNegate")
        return self._make_result_ct(-ct._data, ct._level, ct._slots)

    def EvalNegateInPlace(self, ct: Ciphertext) -> Ciphertext:
        self._verify_leveledshe("EvalNegateInPlace")
        ct._data = -ct._data
        return ct

    def EvalMultMany(self, cts: list) -> Ciphertext:
        self._verify_advancedshe("EvalMultMany")
        result = cts[0].Clone()
        for ct in cts[1:]:
            result = self.EvalMult(result, ct)
        return result

    def EvalAddMany(self, cts: list) -> Ciphertext:
        self._verify_advancedshe("EvalAddMany")
        result = cts[0].Clone()
        for ct in cts[1:]:
            result = self.EvalAdd(result, ct)
        return result

    def EvalAddManyInPlace(self, cts: list) -> Ciphertext:
        self._verify_advancedshe("EvalAddManyInPlace")
        result = cts[0]
        for ct in cts[1:]:
            result = self.EvalAddInPlace(result, ct)
        return result

    # --- Rescale / ModReduce ---
    # In FIXEDMANUAL mode these are the operations that actually consume a level.
    # Each call increments the ciphertext level by 1 and checks against the depth budget.
    def Rescale(self, ct: Ciphertext) -> Ciphertext:
        new_ct = ct.Clone()
        new_ct._level = ct._level + 1
        self._check_level(new_ct._level, "Rescale")
        return new_ct

    def RescaleInPlace(self, ct: Ciphertext) -> Ciphertext:
        ct._level += 1
        self._check_level(ct._level, "RescaleInPlace")
        return ct

    def ModReduce(self, ct: Ciphertext) -> Ciphertext:
        new_ct = ct.Clone()
        new_ct._level = ct._level + 1
        self._check_level(new_ct._level, "ModReduce")
        return new_ct

    def ModReduceInPlace(self, ct: Ciphertext) -> Ciphertext:
        ct._level += 1
        self._check_level(ct._level, "ModReduceInPlace")
        return ct

    def Compress(self, ct: Ciphertext, towers: int = 1) -> Ciphertext:
        return ct.Clone()

    # --- Rotations ---
    def EvalRotate(self, ct: Ciphertext, index: int) -> Ciphertext:
        self._verify_leveledshe("EvalRotate")
        return self._make_result_ct(np.roll(ct._data, -index), ct._level, ct._slots)

    def EvalFastRotationPrecompute(self, ct: Ciphertext) -> Any:
        return ct._data.copy()

    def EvalFastRotation(self, ct, index, ring_dim_or_precomp, precomp_or_none=None) -> Ciphertext:
        return self.EvalRotate(ct, index)

    def EvalFastRotationExt(self, ct, index, precomp, add_first: bool = True) -> Ciphertext:
        return self.EvalRotate(ct, index)

    def EvalAtIndex(self, ct: Ciphertext, index: int) -> Ciphertext:
        return self.EvalRotate(ct, index)

    def EvalAutomorphism(self, ct, index, eval_keys=None) -> Ciphertext:
        return self.EvalRotate(ct, index)

    def EvalSum(self, ct: Ciphertext, batch_size: int) -> Ciphertext:
        self._verify_advancedshe("EvalSum")
        data = ct._data.copy()
        n = len(data)
        effective_batch = min(batch_size, n) if batch_size > 0 else n
        total = np.sum(data[:effective_batch])
        result = np.zeros_like(data)
        result[0] = total
        return self._make_result_ct(result, ct._level, ct._slots)

    def EvalSumRows(self, ct: Ciphertext, row_size: int, eval_keys=None) -> Ciphertext:
        self._verify_advancedshe("EvalSumRows")
        data = ct._data.copy()
        n = len(data)
        result = np.zeros_like(data)
        for i in range(0, n, row_size):
            s = np.sum(data[i:i+row_size])
            result[i] = s
        return self._make_result_ct(result, ct._level, ct._slots)

    def EvalSumCols(self, ct: Ciphertext, row_size: int, eval_keys=None) -> Ciphertext:
        self._verify_advancedshe("EvalSumCols")
        data = ct._data.copy()
        n = len(data)
        num_rows = n // row_size if row_size > 0 else 1
        result = np.zeros_like(data)
        for j in range(row_size):
            s = sum(data[i * row_size + j] for i in range(num_rows) if i * row_size + j < n)
            result[j] = s
        return self._make_result_ct(result, ct._level, ct._slots)

    def EvalInnerProduct(self, a, b, batch_size: int) -> Ciphertext:
        self._verify_advancedshe("EvalInnerProduct")
        da, la, _ = self._coerce_operand(a)
        db, lb, _ = self._coerce_operand(b)
        da, db = self._broadcast(da, db)
        new_level = max(la, lb)  # FIXEDMANUAL: EvalMult no auto-Rescale; no explicit Rescale in EvalInnerProduct
        effective = min(batch_size, len(da)) if batch_size > 0 else len(da)
        total = np.sum(da[:effective] * db[:effective])
        result = np.zeros(len(da), dtype=np.float64)
        result[0] = total
        slots = a._slots if isinstance(a, Ciphertext) else (b._slots if isinstance(b, Ciphertext) else len(da))
        return self._make_result_ct(result, new_level, slots)

    # --- Polynomial / Chebyshev approximations ---
    @staticmethod
    def _effective_degree(coeffs: list) -> int:
        """Return actual degree (index of last non-zero coeff), like OpenFHE's Degree()."""
        d = len(coeffs) - 1
        while d > 0 and abs(coeffs[d]) < 1e-300:
            d -= 1
        return d

    def _poly_depth(self, degree: int) -> int:
        """Multiplicative depth for polynomial evaluation.  For degree >= 5 the
        Paterson-Stockmeyer path in OpenFHE consumes one level fewer than the
        raw table value from GetDepthByDegree."""
        raw = _get_depth_by_degree(degree)
        if degree >= 5:
            return raw - 1
        return raw

    def EvalChebyshevSeries(self, ct: Ciphertext, coeffs: list,
                             a: float, b: float) -> Ciphertext:
        self._verify_advancedshe("EvalChebyshevSeries")
        degree = self._effective_degree(coeffs)
        needs_affine = not (a == -1.0 and b == 1.0)
        levels_consumed = self._poly_depth(degree) + (1 if needs_affine else 0)
        new_level = ct._level + levels_consumed
        self._check_level(new_level, f"EvalChebyshevSeries(degree={degree})")

        x = ct._data.copy()
        mapped = 2.0 * (x - a) / (b - a) - 1.0

        c = np.array(coeffs, dtype=np.float64)
        c[0] /= 2.0
        result = chebyshev.chebval(mapped, c)

        return self._make_result_ct(np.asarray(result, dtype=np.float64), new_level, ct._slots)

    def EvalChebyshevSeriesLinear(self, ct: Ciphertext, coeffs: list,
                                   a: float, b: float) -> Ciphertext:
        return self.EvalChebyshevSeries(ct, coeffs, a, b)

    def EvalChebyshevSeriesPS(self, ct: Ciphertext, coeffs: list,
                               a: float, b: float) -> Ciphertext:
        return self.EvalChebyshevSeries(ct, coeffs, a, b)

    def EvalChebyshevFunction(self, func: Callable, ct: Ciphertext,
                               a: float, b: float, degree: int) -> Ciphertext:
        self._verify_advancedshe("EvalChebyshevFunction")
        needs_affine = not (a == -1.0 and b == 1.0)
        levels_consumed = self._poly_depth(degree) + (1 if needs_affine else 0)
        new_level = ct._level + levels_consumed
        self._check_level(new_level, f"EvalChebyshevFunction(degree={degree})")

        x = ct._data.copy()
        mapped = 2.0 * (x - a) / (b - a) - 1.0

        nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
        values = np.array([func(0.5 * (b - a) * t + 0.5 * (a + b)) for t in nodes])
        coeffs = chebyshev.chebfit(nodes, values, degree)
        result = chebyshev.chebval(mapped, coeffs)

        return self._make_result_ct(np.asarray(result, dtype=np.float64), new_level, ct._slots)

    def EvalPoly(self, ct: Ciphertext, coeffs: list) -> Ciphertext:
        self._verify_advancedshe("EvalPoly")
        degree = self._effective_degree(coeffs)
        levels_consumed = self._poly_depth(degree)
        new_level = ct._level + levels_consumed
        self._check_level(new_level, f"EvalPoly(degree={degree})")

        x = ct._data.copy()
        result = np.polyval(list(reversed(coeffs)), x)
        return self._make_result_ct(np.asarray(result, dtype=np.float64), new_level, ct._slots)

    def EvalPolyLinear(self, ct: Ciphertext, coeffs: list) -> Ciphertext:
        return self.EvalPoly(ct, coeffs)
    def EvalPolyPS(self, ct: Ciphertext, coeffs: list) -> Ciphertext:
        return self.EvalPoly(ct, coeffs)

    def EvalLogistic(self, ct: Ciphertext, a: float, b: float,
                      degree: int) -> Ciphertext:
        self._verify_advancedshe("EvalLogistic")
        levels_consumed = self._poly_depth(degree)
        new_level = ct._level + levels_consumed
        self._check_level(new_level, "EvalLogistic")
        x = ct._data.copy()
        result = 1.0 / (1.0 + np.exp(-x))
        return self._make_result_ct(result, new_level, ct._slots)

    def EvalSin(self, ct: Ciphertext, a: float, b: float,
                 degree: int) -> Ciphertext:
        self._verify_advancedshe("EvalSin")
        levels_consumed = self._poly_depth(degree)
        new_level = ct._level + levels_consumed
        self._check_level(new_level, "EvalSin")
        return self._make_result_ct(np.sin(ct._data), new_level, ct._slots)

    def EvalCos(self, ct: Ciphertext, a: float, b: float,
                 degree: int) -> Ciphertext:
        self._verify_advancedshe("EvalCos")
        levels_consumed = self._poly_depth(degree)
        new_level = ct._level + levels_consumed
        self._check_level(new_level, "EvalCos")
        return self._make_result_ct(np.cos(ct._data), new_level, ct._slots)

    def EvalDivide(self, ct: Ciphertext, a: float, b: float,
                    degree: int) -> Ciphertext:
        self._verify_advancedshe("EvalDivide")
        levels_consumed = self._poly_depth(degree)
        new_level = ct._level + levels_consumed
        self._check_level(new_level, "EvalDivide")
        x = ct._data.copy()
        result = np.where(np.abs(x) > 1e-15, 1.0 / x, 0.0)
        return self._make_result_ct(result, new_level, ct._slots)

    # --- Linear weighted sum ---
    def EvalLinearWSum(self, cts: list, weights: list) -> Ciphertext:
        self._verify_advancedshe("EvalLinearWSum")
        # Each term is ct×scalar — no level consumed (unlike ct×ct)
        result = None
        for ct, w in zip(cts, weights):
            term = self._make_result_ct(ct._data * float(w), ct._level, ct._slots)
            result = term if result is None else self.EvalAdd(result, term)
        # Mirror ckksrns-advancedshe.cpp:138 — unconditional ModReduceInPlace consumes 1 level
        result._level += 1
        self._check_level(result._level, "EvalLinearWSum")
        return result

    def EvalLinearWSumMutable(self, cts: list, weights: list) -> Ciphertext:
        return self.EvalLinearWSum(cts, weights)

    # --- Bootstrapping ---
    def EvalBootstrapSetup(self, level_budget=None, dim1=None, slots=None,
                            correction_factor=None):
        self._verify_fhe("EvalBootstrapSetup")
        if level_budget is not None:
            self._bootstrap_depth = sum(level_budget) if hasattr(level_budget, '__iter__') else level_budget

    def EvalBootstrapKeyGen(self, secret_key, num_slots: int = 0):
        self._verify_fhe("EvalBootstrapKeyGen")

    def EvalBootstrapPrecompute(self, *args, **kwargs):
        self._verify_fhe("EvalBootstrapPrecompute")

    def EvalBootstrap(self, ct: Ciphertext, num_iterations: int = 1,
                       precision: int = 0) -> Ciphertext:
        self._verify_fhe("EvalBootstrap")
        new_ct = ct.Clone()
        new_ct._level = 0
        return new_ct

    # --- Scheme switching stubs (not functional, just prevent crashes) ---
    def EvalCKKStoFHEWSetup(self, *a, **kw): pass
    def EvalCKKStoFHEWKeyGen(self, *a, **kw): pass
    def EvalCKKStoFHEWPrecompute(self, *a, **kw): pass
    def EvalCKKStoFHEW(self, *a, **kw): return []
    def EvalFHEWtoCKKSSetup(self, *a, **kw): pass
    def EvalFHEWtoCKKSKeyGen(self, *a, **kw): pass
    def EvalFHEWtoCKKS(self, *a, **kw): return Ciphertext()
    def EvalSchemeSwitchingSetup(self, *a, **kw): pass
    def EvalSchemeSwitchingKeyGen(self, *a, **kw): pass
    def EvalCompareSwitchPrecompute(self, *a, **kw): pass
    def EvalCompareSchemeSwitching(self, *a, **kw): return (Ciphertext(), Ciphertext())
    def EvalMinSchemeSwitching(self, *a, **kw): return (Ciphertext(), Ciphertext())
    def EvalMinSchemeSwitchingAlt(self, *a, **kw): return (Ciphertext(), Ciphertext())
    def EvalMaxSchemeSwitching(self, *a, **kw): return (Ciphertext(), Ciphertext())
    def EvalMaxSchemeSwitchingAlt(self, *a, **kw): return (Ciphertext(), Ciphertext())

    # --- Key cache stubs ---
    def FindAutomorphismIndex(self, idx): return idx
    def FindAutomorphismIndices(self, indices): return indices
    def GetEvalSumKeyMap(self, tag=""): return {}
    def GetBinCCForSchemeSwitch(self): return BinFHEContext()

    @staticmethod
    def InsertEvalSumKey(keys, tag=""): pass
    @staticmethod
    def InsertEvalMultKey(keys): pass
    @staticmethod
    def InsertEvalAutomorphismKey(keys, tag=""): pass
    @staticmethod
    def ClearEvalAutomorphismKeys(): pass

    def GetEvalMultKeyVector(self, tag=""): return []
    def GetEvalAutomorphismKeyMap(self, tag=""): return {}

    # --- PRE stubs ---
    def ReKeyGen(self, old_key, new_key): return EvalKey()
    def ReEncrypt(self, ct, eval_key, pk=None): return ct.Clone()

    # --- Serialization stubs (no-ops) ---
    def SerializeEvalMultKey(self, *a, **kw): return True
    def DeserializeEvalMultKey(self, *a, **kw): return True
    def SerializeEvalAutomorphismKey(self, *a, **kw): return True
    def DeserializeEvalAutomorphismKey(self, *a, **kw): return True

    # --- Multiparty stubs ---
    def MultipartyKeyGen(self, *a, **kw): return KeyPair()
    def MultipartyDecryptLead(self, *a, **kw): return Plaintext(np.zeros(1))
    def MultipartyDecryptMain(self, *a, **kw): return Plaintext(np.zeros(1))
    def MultipartyDecryptFusion(self, *a, **kw): return Plaintext(np.zeros(1))
    def MultiKeySwitchGen(self, *a, **kw): return EvalKey()
    def MultiEvalAtIndexKeyGen(self, *a, **kw): return {}
    def MultiEvalSumKeyGen(self, *a, **kw): return {}
    def MultiAddEvalAutomorphismKeys(self, *a, **kw): pass
    def MultiAddPubKeys(self, *a, **kw): return PublicKey()
    def MultiAddEvalKeys(self, *a, **kw): return EvalKey()
    def MultiAddEvalMultKeys(self, *a, **kw): return EvalKey()
    def MultiMultEvalKey(self, *a, **kw): return EvalKey()
    def MultiAddEvalSumKeys(self, *a, **kw): return {}
    def EvalMerge(self, *a, **kw): return Ciphertext()

    def __repr__(self):
        return (f"CryptoContext(sim, depth={self._mult_depth}, "
                f"batch={self._batch_size}, ring={self._ring_dim})")


# ---------------------------------------------------------------------------
# FHECKKSRNS helper
# ---------------------------------------------------------------------------

class FHECKKSRNS:
    """Replicates OpenFHE's FHECKKSRNS static helpers for bootstrap depth."""
    _R_UNIFORM = 6
    _R_SPARSE = 3
    _MOD_DEPTH_SPARSE = _get_depth_by_degree(44) + _R_SPARSE    # 7 + 3 = 10
    _MOD_DEPTH_UNIFORM = _get_depth_by_degree(88) + _R_UNIFORM   # 8 + 6 = 14

    def __init__(self):
        pass

    @staticmethod
    def _get_mod_depth_internal(secret_key_dist) -> int:
        if int(secret_key_dist) == int(SecretKeyDist.UNIFORM_TERNARY):
            return FHECKKSRNS._MOD_DEPTH_UNIFORM
        return FHECKKSRNS._MOD_DEPTH_SPARSE

    @staticmethod
    def GetBootstrapDepth(approx_mod_depth_or_level_budget, level_budget_or_secret_key_dist,
                           secret_key_dist=None) -> int:
        if secret_key_dist is not None:
            approx_mod_depth = approx_mod_depth_or_level_budget
            level_budget = level_budget_or_secret_key_dist
            if int(secret_key_dist) == int(SecretKeyDist.UNIFORM_TERNARY):
                approx_mod_depth += FHECKKSRNS._R_UNIFORM - 1
            return approx_mod_depth + sum(level_budget)
        level_budget = approx_mod_depth_or_level_budget
        skd = level_budget_or_secret_key_dist
        approx_mod_depth = FHECKKSRNS._get_mod_depth_internal(skd)
        return approx_mod_depth + sum(level_budget)


# ---------------------------------------------------------------------------
# Module-level factory functions
# ---------------------------------------------------------------------------

_all_contexts: list[CryptoContext] = []

def GenCryptoContext(params) -> CryptoContext:
    cc = CryptoContext(params)
    _all_contexts.append(cc)
    return cc

def GetAllContexts() -> list:
    return list(_all_contexts)

def ReleaseAllContexts():
    _all_contexts.clear()

def ClearEvalMultKeys():
    pass

# Module-level Chebyshev helpers
def EvalChebyshevCoefficients(func: Callable, a: float, b: float, degree: int) -> list:
    nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
    x_vals = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    y_vals = np.array([func(x) for x in x_vals])
    coeffs = chebyshev.chebfit(nodes, y_vals, degree)
    return coeffs.tolist()

def EvalChebyshevFunctionPtxt(func: Callable, x: list, a: float, b: float,
                               degree: int) -> list:
    arr = np.array(x, dtype=np.float64)
    mapped = 2.0 * (arr - a) / (b - a) - 1.0
    nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
    x_vals = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    y_vals = np.array([func(xv) for xv in x_vals])
    coeffs = chebyshev.chebfit(nodes, y_vals, degree)
    result = chebyshev.chebval(mapped, coeffs)
    return result.tolist()


# Serialization stubs
def SerializeToFile(*args, **kwargs): return True
def Serialize(*args, **kwargs): return ("", True)
def DeserializeCryptoContext(*args, **kwargs): return (None, False)
def DeserializeCryptoContextString(*args, **kwargs): return (None, False)
def DeserializePublicKey(*args, **kwargs): return (None, False)
def DeserializePublicKeyString(*args, **kwargs): return (None, False)
def DeserializePrivateKey(*args, **kwargs): return (None, False)
def DeserializePrivateKeyString(*args, **kwargs): return (None, False)
def DeserializeCiphertext(*args, **kwargs): return (None, False)
def DeserializeCiphertextString(*args, **kwargs): return (None, False)
def DeserializeEvalKey(*args, **kwargs): return (None, False)
def DeserializeEvalKeyString(*args, **kwargs): return (None, False)
def DeserializeEvalKeyMap(*args, **kwargs): return (None, False)
def DeserializeEvalKeyMapString(*args, **kwargs): return (None, False)
def DeserializeEvalKeyMapVectorString(*args, **kwargs): return (None, False)
def DeserializeEvalMultKeyString(*args, **kwargs): return (None, False)
def DeserializeEvalAutomorphismKeyString(*args, **kwargs): return (None, False)
def SerializeEvalMultKeyString(*args, **kwargs): return ("", True)
def SerializeEvalAutomorphismKeyString(*args, **kwargs): return ("", True)
def EnablePrecomputeCRTTablesAfterDeserializaton(): pass
def DisablePrecomputeCRTTablesAfterDeserializaton(): pass
