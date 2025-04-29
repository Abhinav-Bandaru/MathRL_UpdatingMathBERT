# train.py

import torch
from mathbert_encoder import MathBERTEncoder
from retriever_cosine import retrieve_top_k_cosine, retrieve_sample_k_cosine
from response_sampler import sample_responses_per_demo
from reward_aggregator import compute_demo_accuracy
from icl_model_wrapper import OpenAIICLModel
from grpo_optimizer import grpo_step  # assume we'll implement this
from datasets import load_dataset
from dotenv import load_dotenv
import os
load_dotenv()


# === Settings ===
API_KEY = os.getenv("OPENAI_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# EMBEDDING_DIM = 768
K = 3
NUM_SAMPLES_PER_DEMO = 5
LEARNING_RATE = 1e-5
MAX_STEPS = 100
TEMPERATURE = 0.7

# === Initialize components ===
encoder = MathBERTEncoder(device=DEVICE, trainable=True)
encoder.train()

icl_model = OpenAIICLModel(api_key=API_KEY, model_name="gpt-4.1-nano", temperature=TEMPERATURE)

optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

# === Mock GSM8K data ===
# gsm8k_data = [
#     {"question": "Maila read 12 x 2 = ... #### 42", "answer": "#### 42"},
#     {"question": "Natalia sold 48/2 = ... #### 72", "answer": "#### 72"},
#     {"question": "Betty has 100 / 2 = ... #### 5", "answer": "#### 5"},
# ]

gsm8k_data = load_dataset('gsm8k', 'train')
gsm8k_data = gsm8k_data[:200]

# === Training Loop ===
for step in range(MAX_STEPS):
    print(f"\n=== Training Step {step+1} ===")

    # Loop through each example as inference
    for inference_index in range(len(gsm8k_data)):
        inference_item = gsm8k_data[inference_index]
        demo_pool = [d for idx, d in enumerate(gsm8k_data) if idx != inference_index]

        Q_inf = inference_item["question"]
        A_gt = inference_item["answer"]
        demos = [(d["question"], d["answer"]) for d in demo_pool]

        # Step 1: Encode inference and demos
        q_emb = encoder.encode([Q_inf], detach=False).squeeze(0)  # (hidden_dim,)
        demo_embs = encoder.encode([q for (q, a) in demos], detach=False)  # (num_demos, hidden_dim)

        # Step 2: Retrieve top-k demos
        # top_k_indices, similarities = retrieve_top_k_cosine(q_emb, demo_embs, k=min(K, len(demos)))
        top_k_indices, similarities = retrieve_sample_k_cosine(q_emb, demo_embs, k=min(K, len(demos)))

        selected_demos = [demos[i] for i in top_k_indices]

        # Step 3: For each selected demo, sample completions
        all_responses = sample_responses_per_demo(
            demo_tuples=selected_demos,
            Q_inf=Q_inf,
            icl_model=icl_model,
            num_samples=NUM_SAMPLES_PER_DEMO
        )

        # Step 4: Compute rewards
        rewards = []
        for responses in all_responses:
            reward = compute_demo_accuracy(responses, A_gt)
            rewards.append(reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)  # (k,)

        # Step 5: GRPO optimization step
        selected_similarities = similarities[top_k_indices]  # (k,)
        loss = grpo_step(
            rewards,
            selected_similarities,
            q_emb,
            None,
            optimizer
        )

        print(f"  Inference on example {inference_index} | Loss: {loss:.4f}")
