"""Format-Preserving Encryption (FF3-1 / NIST SP 800-38G) for test metrics
and for hyperparameters broadcast from server to clients alongside the
global model update."""
from typing import Dict

from ff3 import FF3Cipher

import config


class FPEEngine:
    def __init__(self, key: str = config.FPE_KEY, tweak: str = config.FPE_TWEAK):
        self.cipher = FF3Cipher.withCustomAlphabet(key, tweak, "0123456789") \
            if hasattr(FF3Cipher, "withCustomAlphabet") else FF3Cipher(key, tweak)

    def encrypt_metric(self, value: float, precision: int = 6) -> str:
        """0.952341 -> '952341' -> FPE-encrypt -> e.g. '418732' (same digit
        format). FF3-1 requires a minimum input length of 6 digits."""
        precision = max(precision, 6)
        digit_str = f"{value:.{precision}f}".split(".")[1]  # digits after decimal
        digit_str = digit_str.zfill(precision)
        return self.cipher.encrypt(digit_str)

    def decrypt_metric(self, ciphertext: str, precision: int = 6) -> float:
        precision = max(precision, 6)
        plain_digits = self.cipher.decrypt(ciphertext)
        return float(f"0.{plain_digits.zfill(precision)}")

    # -- hyperparameter broadcast encryption --------------------------------
    def encrypt_hyperparams(self, learning_rate: float, local_epochs: int) -> Dict[str, str]:
        """Encrypts the client-training hyperparameters (learning_rate,
        local_epochs) that the server sends alongside each round's global
        model update, so they aren't visible in plaintext on the wire.
        Both are encoded as zero-padded 6-digit strings (FF3-1's minimum
        input length) before FPE encryption.
        """
        lr_digits = str(int(round(learning_rate * 1_000_000))).zfill(6)
        epochs_digits = str(int(local_epochs)).zfill(6)
        return {
            "learning_rate": self.cipher.encrypt(lr_digits),
            "local_epochs": self.cipher.encrypt(epochs_digits),
        }

    def decrypt_hyperparams(self, encrypted: Dict[str, str]) -> Dict[str, float]:
        lr_digits = self.cipher.decrypt(encrypted["learning_rate"])
        epochs_digits = self.cipher.decrypt(encrypted["local_epochs"])
        return {
            "learning_rate": int(lr_digits) / 1_000_000,
            "local_epochs": int(epochs_digits),
        }


if __name__ == "__main__":
    engine = FPEEngine()
    original = 0.9523
    enc = engine.encrypt_metric(original)
    dec = engine.decrypt_metric(enc)
    print(f"original={original} -> encrypted='{enc}' -> decrypted={dec}")
    assert abs(dec - original) < 1e-6, "FPE roundtrip mismatch!"
    print("FPE metric roundtrip OK")

    hp_enc = engine.encrypt_hyperparams(learning_rate=0.05, local_epochs=3)
    hp_dec = engine.decrypt_hyperparams(hp_enc)
    print(f"hyperparams encrypted={hp_enc} -> decrypted={hp_dec}")
    assert abs(hp_dec["learning_rate"] - 0.05) < 1e-9
    assert hp_dec["local_epochs"] == 3
    print("FPE hyperparameter roundtrip OK")
