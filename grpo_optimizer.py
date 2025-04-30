import torch
# def grpo_step(rewards, logp, q_emb, q_ref, optimizer, beta=0.04):
#     adv     = (rewards - rewards.mean()) / (rewards.std() + 1e-8)         # Â_t
#     policy_loss = -(adv.detach() * logp.squeeze(1)).sum()

#     # kl = ((q_emb.detach() - q_ref)**2).sum()              
#     loss = policy_loss #+ beta * kl
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     return loss


def grpo_step(rewards, logp, q_emb, demo_embs, optimizer, beta=0.04):
    """
    rewards: 1-D   tensor  (k,)        – r_t   after sampling
    logp   : 1-D   tensor  (k,)        – log π(a_t)
    """
    # normalise rewards inside the k-set  → Â_t
    std = rewards.std() + (1e-5 if rewards.std() < 1e-5 else 0)
    print("Rewards: ", rewards)
    adv = (rewards - rewards.mean()) / std # (k,)
    print("Advantages: ", adv)
    # shapes must match: both (k,)
    policy_loss = -(adv.detach() * logp).mean()
    print("Policy Loss: ", policy_loss)
    # print(logp)
    # optional KL on the query embedding
    # kl = (q_emb.detach() - q_ref).pow(2).sum()

    # high_reward_threshold = 1
    # high_reward_demo_inds = (adv > high_reward_threshold).nonzero(as_tuple=True)[0]

    # if len(high_reward_demo_inds) > 0:
    #     high_reward_demo_embs = demo_embs[high_reward_demo_inds]
    #     q_emb_expanded = q_emb.expand_as(high_reward_demo_embs)
    #     embedding_loss = torch.nn.functional.mse_loss(high_reward_demo_embs, q_emb_expanded)

    #     loss = policy_loss + embedding_loss
    # else:
    #     loss = policy_loss
    
    loss = policy_loss # +  beta * kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()