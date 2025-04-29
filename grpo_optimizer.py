
# def grpo_step(rewards, logp, q_emb, q_ref, optimizer, beta=0.04):
#     adv     = (rewards - rewards.mean()) / (rewards.std() + 1e-8)         # Â_t
#     policy_loss = -(adv.detach() * logp.squeeze(1)).sum()

#     # kl = ((q_emb.detach() - q_ref)**2).sum()              
#     loss = policy_loss #+ beta * kl
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     return loss


def grpo_step(rewards, logp, q_emb, q_ref, optimizer, beta=0.04):
    """
    rewards: 1-D   tensor  (k,)        – r_t   after sampling
    logp   : 1-D   tensor  (k,)        – log π(a_t)
    """
    # normalise rewards inside the k-set  → Â_t
    std = rewards.std() + (1e-4 if rewards.std() < 1e-4 else 0)
    adv = (rewards - rewards.mean()) / std # (k,)
    print(adv)
    print(std)
    print(logp)
    # shapes must match: both (k,)
    policy_loss = -(adv.detach() * logp).mean()

    # optional KL on the query embedding
    # kl = (q_emb.detach() - q_ref).pow(2).sum()

    loss = policy_loss # +  beta * kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()