# response_sampler.py
import torch 
from concurrent.futures import ThreadPoolExecutor, as_completed

def _call_generate(icl_model, prompt):
    """Thin wrapper so we can pass just one arg to the executor."""
    output = icl_model.generate(prompt)
    print("OUTPUT: ", output)
    # return output[0].answer  # we only need `output`, ignore logprob
    return output[0]

def sample_responses_per_demo(demo_tuples, Q_inf, icl_model, num_samples=1, parallel=False):
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
    if parallel:
        # how many parallel requests you want in flight
        PARALLEL_WORKERS = 8          # tune to your rate-limit


        all_responses = []

        for i, (demo_q, demo_a) in enumerate(demo_tuples):
            prompt = f"Q: {demo_q}\nA: {demo_a}\n\nQ: {Q_inf}\nA:"
            completions = []

            if parallel and num_samples > 1:
                with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
                    futures = [pool.submit(_call_generate, icl_model, prompt)
                            for _ in range(num_samples)]
                    completions = [f.result() for f in futures]      # preserves order
            else:
                for _ in range(num_samples):
                    output, _ = icl_model.generate(prompt)
                    print("OUTPUT: ", output)
                    completions.append(output.answer)

            all_responses.append(completions)
            if i == 0:
                print("------------Demo 1--------")
                print(prompt)
                print(completions[0])
        return all_responses 
    
    all_responses = []

    for demo_q, demo_a in demo_tuples:
        prompt = f"Q: {demo_q}\nA: {demo_a}\n\nQ: {Q_inf}\nA:"
        completions = []

        for i in range(num_samples):
            output, _ = icl_model.generate(prompt)
            # if i == 0:
            #     print(f"---------------{i}-----------")
            #     print(prompt)
            #     print(output)
            completions.append(output)

        all_responses.append(completions)

    return all_responses
