# retriever_cosine.py

import torch
import torch.nn.functional as F

def retrieve_top_k_cosine(query_embedding, demo_embeddings, k):
    """
    Retrieve top-k demos based on cosine similarity with the query.

    Args:
        query_embedding (Tensor): shape (hidden_dim,)
        demo_embeddings (Tensor): shape (num_demos, hidden_dim)
        k (int): number of top demos to retrieve

    Returns:
        top_k_indices (list[int]): indices of top-k demos by similarity
        similarities (Tensor): shape (num_demos,) — cosine similarity scores
    """
    # Compute cosine similarity between query and all demos
    similarities = F.cosine_similarity(query_embedding.unsqueeze(0), demo_embeddings, dim=1)
    
    # Get indices of top-k demos
    top_k = torch.topk(similarities, k=k)
    top_k_indices = top_k.indices.tolist()

    return top_k_indices, similarities  # return full similarity tensor if needed for GRPO

def retrieve_sample_k_cosine(q_emb: torch.Tensor,
                             pool_emb: torch.Tensor,
                             k: int,
                             tau: float = 0.07):
    """
    Returns
    -------
    idx_set : LongTensor (k,)        indices of the k chosen examples
    logp    : FloatTensor (k,)        log πθ(a_t | s_t) for each chosen index
    """
    # 0. L2-NORMALISE both query and candidates
    # q_emb    = F.normalize(q_emb,    p=2, dim=-1)          # (D,)  or (B,D)
    # pool_emb = F.normalize(pool_emb, p=2, dim=-1)          # (N,D)
    
    # 1. cosine-similarity logits
    logits = (q_emb @ pool_emb.T) / tau    # shape (N,)

    # 2. differentiable Gumbel-softmax draw (probabilities, not indices)
    #    hard=False keeps it a soft prob vector so gradients flow
    probs = F.gumbel_softmax(logits, tau=1.0, hard=False)  # shape (N,)

    # 3. choose the k highest-prob items   (no replacement)
    idx_set = probs.topk(k).indices                     # (k,)

    # 4. log-probabilities for the sampled actions
    logp_all = torch.log_softmax(logits, dim=-1)        # (N,)
    logp = logp_all[idx_set]                            # (k,)

    return idx_set, logp