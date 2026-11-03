"""
CKKS homomorphic encryption for federated GraphSAGE weight updates.

Backend: Pyfhel (native Windows build via MSVC C++ Build Tools). Model
weight tensors are flattened, chunked to CKKS_SLOTS-sized vectors, and each
chunk is independently encrypted. Aggregation is done via homomorphic
addition directly on ciphertexts — the server never sees plaintext client
updates.

If Pyfhel's C++ build tools aren't available on Windows, set
config.CKKS_BACKEND = "tenseal" and run inside WSL2 (TenSEAL ships Linux
wheels only).
"""
from typing import Dict, List, Tuple

import numpy as np
import torch

import config


class CKKSEngine:
    def __init__(self):
        self.backend = config.CKKS_BACKEND
        if self.backend == "pyfhel":
            from Pyfhel import Pyfhel

            self.HE = Pyfhel()
            self.HE.contextGen(
                scheme="CKKS",
                n=config.POLY_MODULUS_DEGREE,
                scale=config.CKKS_SCALE,
                qi_sizes=config.QI_SIZES,
            )
            self.HE.keyGen()
            self.HE.relinKeyGen()
        elif self.backend == "tenseal":
            import tenseal as ts

            self.ts = ts
            self.HE = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=config.POLY_MODULUS_DEGREE,
                coeff_mod_bit_sizes=config.QI_SIZES,
            )
            self.HE.global_scale = config.CKKS_SCALE
            self.HE.generate_galois_keys()
        else:
            raise ValueError(f"Unknown CKKS_BACKEND: {self.backend}")

        self.slots = config.CKKS_SLOTS

    # -- flatten/unflatten model state_dicts -------------------------------
    @staticmethod
    def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, List[Tuple[str, torch.Size]]]:
        shapes = []
        flats = []
        for name, tensor in state_dict.items():
            shapes.append((name, tensor.shape))
            flats.append(tensor.detach().cpu().numpy().flatten())
        return np.concatenate(flats), shapes

    @staticmethod
    def unflatten_to_state_dict(flat: np.ndarray, shapes: List[Tuple[str, torch.Size]]) -> Dict[str, torch.Tensor]:
        state_dict = {}
        offset = 0
        for name, shape in shapes:
            numel = int(np.prod(shape))
            chunk = flat[offset : offset + numel].reshape(shape)
            state_dict[name] = torch.tensor(chunk, dtype=torch.float32)
            offset += numel
        return state_dict

    # -- encrypt / aggregate / decrypt --------------------------------------
    def encrypt_vector(self, vec: np.ndarray) -> List:
        """Chunks `vec` into CKKS_SLOTS-sized pieces and encrypts each."""
        chunks = [vec[i : i + self.slots] for i in range(0, len(vec), self.slots)]
        ciphertexts = []
        for chunk in chunks:
            padded = np.zeros(self.slots)
            padded[: len(chunk)] = chunk
            if self.backend == "pyfhel":
                ct = self.HE.encryptFrac(padded)
            else:  # tenseal
                ct = self.ts.ckks_vector(self.HE, padded.tolist())
            ciphertexts.append(ct)
        return ciphertexts

    def aggregate_encrypted(self, client_ciphertexts: List[List]) -> List:
        """Homomorphic addition across all clients' encrypted vectors,
        then scalar-multiplies by 1/N for the federated average."""
        n_clients = len(client_ciphertexts)
        n_chunks = len(client_ciphertexts[0])
        aggregated = []
        for chunk_idx in range(n_chunks):
            acc = client_ciphertexts[0][chunk_idx]
            for cid in range(1, n_clients):
                acc = acc + client_ciphertexts[cid][chunk_idx]
            acc = acc * (1.0 / n_clients)
            aggregated.append(acc)
        return aggregated

    def decrypt_vector(self, ciphertexts: List, original_length: int) -> np.ndarray:
        pieces = []
        for ct in ciphertexts:
            if self.backend == "pyfhel":
                pieces.append(self.HE.decryptFrac(ct))
            else:
                pieces.append(np.array(ct.decrypt()))
        flat = np.concatenate(pieces)[:original_length]
        return flat


if __name__ == "__main__":
    from models.graphsage_model import GraphSAGENet

    engine = CKKSEngine()
    m1 = GraphSAGENet()
    m2 = GraphSAGENet()

    flat1, shapes = engine.flatten_state_dict(m1.state_dict())
    flat2, _ = engine.flatten_state_dict(m2.state_dict())

    ct1 = engine.encrypt_vector(flat1)
    ct2 = engine.encrypt_vector(flat2)

    agg = engine.aggregate_encrypted([ct1, ct2])
    decrypted = engine.decrypt_vector(agg, len(flat1))

    expected = (flat1 + flat2) / 2
    err = np.abs(decrypted - expected).max()
    print(f"CKKS roundtrip max abs error: {err:.6f} (expect < 0.01)")
