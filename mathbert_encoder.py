# mathbert_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F          #  <- add this
from transformers import BertTokenizer, BertModel

class MathBERTEncoder(nn.Module):
    def __init__(self, model_name="tbs17/MathBERT", device="cuda", trainable=True):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = device
        self.trainable = trainable

        self.model.to(self.device)
        if not trainable:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def encode(self, texts, max_length=512, detach=False, normalize=True):
        """
        Encode a list of texts into MathBERT embeddings.

        Args:
            texts (list[str]): list of strings to encode
            max_length (int): max sequence length
            detach (bool): if True, detaches output from gradient flow. True during inference, False during training

        Returns:
            Tensor: shape (batch_size, hidden_size)
        """
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)

        output = self.model(**encoded_input)
        hidden_states = output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        cls_embeddings = hidden_states[:, 0, :]   # (batch_size, hidden_size)
        if normalize:
            # safe L2 normalisation: ‖v‖₂ = √(Σ v_i²) ; clamp to avoid 0
            cls_embeddings = F.normalize(cls_embeddings, p=2, dim=-1, eps=1e-6)
        if detach:
            cls_embeddings = cls_embeddings.detach()

        return cls_embeddings  # (batch_size, hidden_size)
    

    def batched_encode(self, texts, batch_size=16, **kwargs):
        """
        Encode a large number of texts in memory-safe batches.

        Args:
            texts (list[str]): input strings
            batch_size (int): how many to encode at once
            kwargs: forwarded to self.encode()

        Returns:
            Tensor: shape (len(texts), hidden_dim)
        """
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                embs = self.encode(batch, **kwargs).cpu()  # ← move to CPU
                all_embeddings.append(embs)

            return torch.cat(all_embeddings, dim=0).to(self.device)  # final tensor back on GPU


