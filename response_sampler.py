# response_sampler.py
import torch 

def sample_responses_per_demo(demo_tuples, Q_inf, icl_model, num_samples=1):
    """
    Generate completions from OpenAI model using (demo + Q_inf) prompts.

    Args:
        demo_tuples (list of (str, str)): each demo is a (question, answer) pair
        Q_inf (str): inference-time question
        icl_model: instance of OpenAIICLModel
        num_samples (int): how many com pletions to generate per demo

    Returns:
        results (list[list[str]]): list of list of responses for each demo
    """
    all_responses = []

    for demo_q, demo_a in demo_tuples:
        prompt = f"Q: {demo_q}\nA: {demo_a}\n\nQ: {Q_inf}\nA:"
        completions = []

        for _ in range(num_samples):
            output, _ = icl_model.generate(prompt)
            completions.append(output)

        all_responses.append(completions)

    return all_responses
