"""
embeddings.py

Pre-trained embedding extraction for Drug-Target Interaction prediction.

Drug   -> ChemBERTa  (seyonec/ChemBERTa-zinc-base-v1)
           Trained on 77M SMILES from ZINC database
           Output: 768-dim representation

Protein -> ESM2       (facebook/esm2_t6_8M_UR50D)
           Meta's protein language model, trained on 250M sequences
           Output: 320-dim representation

Embeddings are extracted once and cached to disk.
Subsequent runs load from cache instantly.
"""

import logging
import numpy as np
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Model identifiers
# ------------------------------------------------------------------
CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
ESM2_MODEL      = "facebook/esm2_t6_8M_UR50D"

# Max token lengths (to prevent OOM on very long inputs)
DRUG_MAX_LEN    = 512
PROTEIN_MAX_LEN = 1024   # ESM2 hard limit; sequences truncated at 1022 AAs


# ------------------------------------------------------------------
# Drug Embedder
# ------------------------------------------------------------------

class DrugEmbedder:
    """
    Extracts drug embeddings using ChemBERTa.
    Mean-pools last hidden states over all non-padding tokens.
    """

    def __init__(self, device: torch.device):
        self.device = device
        logger.info(f"Loading ChemBERTa  [{CHEMBERTA_MODEL}]  ...")
        self.tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
        self.model     = AutoModel.from_pretrained(CHEMBERTA_MODEL).to(device)
        self.model.eval()
        self.embed_dim = self.model.config.hidden_size
        logger.info(f"ChemBERTa ready — embedding dim: {self.embed_dim}")

    @torch.no_grad()
    def embed(self, smiles_list, batch_size: int = 64) -> np.ndarray:
        """
        Parameters
        ----------
        smiles_list : list[str]
        batch_size  : int

        Returns
        -------
        np.ndarray  shape (N, embed_dim)
        """
        all_embs = []

        for start in tqdm(range(0, len(smiles_list), batch_size),
                          desc="Drug  embeddings", unit="batch", ncols=80):
            batch   = [str(s) for s in smiles_list[start: start + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=DRUG_MAX_LEN,
                return_tensors="pt",
            ).to(self.device)

            out  = self.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(emb.cpu().float().numpy())

        return np.vstack(all_embs)


# ------------------------------------------------------------------
# Protein Embedder
# ------------------------------------------------------------------

class ProteinEmbedder:
    """
    Extracts protein embeddings using ESM2.
    Mean-pools last hidden states over amino acid positions
    (excluding [CLS] and [EOS] special tokens).
    """

    def __init__(self, device: torch.device):
        self.device = device
        logger.info(f"Loading ESM2  [{ESM2_MODEL}]  ...")
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
        self.model     = AutoModel.from_pretrained(ESM2_MODEL).to(device)
        self.model.eval()
        self.embed_dim = self.model.config.hidden_size
        logger.info(f"ESM2 ready — embedding dim: {self.embed_dim}")

    @torch.no_grad()
    def embed(self, sequences, batch_size: int = 16) -> np.ndarray:
        """
        Parameters
        ----------
        sequences  : list[str]  — single-letter AA sequences
        batch_size : int        — smaller than drug batch due to longer sequences

        Returns
        -------
        np.ndarray  shape (N, embed_dim)
        """
        # Truncate sequences that exceed ESM2 max length
        max_aa = PROTEIN_MAX_LEN - 2     # reserve 2 tokens for [CLS] and [EOS]
        seqs   = [str(s)[:max_aa] for s in sequences]
        all_embs = []

        for start in tqdm(range(0, len(seqs), batch_size),
                          desc="Protein embeddings", unit="batch", ncols=80):
            batch   = seqs[start: start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=PROTEIN_MAX_LEN,
                return_tensors="pt",
            ).to(self.device)

            out  = self.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(emb.cpu().float().numpy())

        return np.vstack(all_embs)


# ------------------------------------------------------------------
# Convenience function: extract + cache both embeddings
# ------------------------------------------------------------------

def get_embeddings(df, device: torch.device, cache_dir: Path) -> tuple:
    """
    Extract ChemBERTa drug embeddings and ESM2 protein embeddings.
    Results are cached to ``cache_dir`` so repeat runs are instant.

    Returns
    -------
    drug_embs    : np.ndarray  (N, drug_dim)
    protein_embs : np.ndarray  (N, protein_dim)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    drug_cache    = cache_dir / "drug_embeddings.npy"
    protein_cache = cache_dir / "protein_embeddings.npy"

    # ---- Drug embeddings ----
    if drug_cache.exists():
        logger.info(f"Loading cached drug embeddings from {drug_cache}")
        drug_embs = np.load(drug_cache)
    else:
        embedder  = DrugEmbedder(device)
        drug_embs = embedder.embed(df["smiles"].tolist())
        np.save(drug_cache, drug_embs)
        logger.info(f"Saved drug embeddings -> {drug_cache}")
        del embedder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Protein embeddings ----
    if protein_cache.exists():
        logger.info(f"Loading cached protein embeddings from {protein_cache}")
        protein_embs = np.load(protein_cache)
    else:
        embedder     = ProteinEmbedder(device)
        protein_embs = embedder.embed(df["protein_sequence"].tolist())
        np.save(protein_cache, protein_embs)
        logger.info(f"Saved protein embeddings -> {protein_cache}")
        del embedder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(
        f"Embeddings ready — drug: {drug_embs.shape}, protein: {protein_embs.shape}"
    )
    return drug_embs, protein_embs
