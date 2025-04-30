# # response_sampler.py
# import torch 
# from concurrent.futures import ThreadPoolExecutor, as_completed

# def _call_generate(icl_model, prompt):
#     """Thin wrapper so we can pass just one arg to the executor."""
#     output = icl_model.generate(prompt)
#     print("OUTPUT: ", output)
#     # return output[0].answer  # we only need `output`, ignore logprob
#     return output[0]

# def sample_responses_per_demo(demo_tuples, Q_inf, icl_model, num_samples=1, parallel=False):
#     """
#     Generate completions from OpenAI model using (demo + Q_inf) prompts.

#     Args:
#         demo_tuples (list of (str, str)): each demo is a (question, answer) pair
#         Q_inf (str): inference-time question
#         icl_model: instance of OpenAIICLModel
#         num_samples (int): how many com pletions to generate per demo

#     Returns:
#         results (list[list[str]]): list of list of responses for each demo
#     """
#     if parallel:
#         # how many parallel requests you want in flight
#         PARALLEL_WORKERS = 8          # tune to your rate-limit


#         all_responses = []

#         for i, (demo_q, demo_a) in enumerate(demo_tuples):
#             prompt = f"Q: {demo_q}\nA: {demo_a}\n\nQ: {Q_inf}\nA:"
#             completions = []

#             if parallel and num_samples > 1:
#                 with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
#                     futures = [pool.submit(_call_generate, icl_model, prompt)
#                             for _ in range(num_samples)]
#                     completions = [f.result() for f in futures]      # preserves order
#             else:
#                 for _ in range(num_samples):
#                     output, _ = icl_model.generate(prompt)
#                     print("OUTPUT: ", output)
#                     completions.append(output.answer)

#             all_responses.append(completions)
#             if i == 0:
#                 print("------------Demo 1--------")
#                 print(prompt)
#                 print(completions[0])
#         return all_responses 
    
#     all_responses = []

#     for demo_q, demo_a in demo_tuples:
#         prompt = f"Q: {demo_q}\nA: {demo_a}\n\nQ: {Q_inf}\nA:"
#         completions = []

#         for i in range(num_samples):
#             output, _ = icl_model.generate(prompt)
#             # if i == 0:
#             #     print(f"---------------{i}-----------")
#             #     print(prompt)
#             #     print(output)
#             completions.append(output)

#         all_responses.append(completions)

#     return all_responses



import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

def _format_demo_prompt(demo_q, demo_a_reasoning, demo_a_answer):
    """
    Format the few-shot examples and inference question.
    """
    formatted_demo = f"""Question: {demo_q}\nSolution: {{"reasoning": "{demo_a_reasoning}", "answer": {demo_a_answer}}}"""
    return formatted_demo

def _call_generate_batch(j, demo_q, demo_solution, demo_answer, Q_inf, icl_model):
    """
    Used internally to run a generation call in parallel.
    """
    demos = _format_demo_prompt(demo_q, demo_solution, demo_answer)
    completion, _ = icl_model.generate(Q_inf, demos)
    return demo_q, completion.answer

# def _call_generate_batch(demo_q, demo_a, Q_inf, icl_model):
#     """Generates a single sample for a given (demo, Q_inf) pair."""
#     prompt = f"Q: {demo_q}\nA: {demo_a}\n\nQ: {Q_inf}\nA:"
#     output, _ = icl_model.generate(prompt)
#     return (demo_q, output.answer)  # returning demo_q for grouping

def sample_responses_per_demo(demo_tuples, Q_inf, icl_model, num_samples=1, parallel=False, max_workers=8):
    """
    Generate completions for (demo + Q_inf) pairs using OpenAI.

    Returns:
        List[List[str]]: each sublist contains `num_samples` completions for one demo
    """
    # if not parallel:
    #     all_responses = []
    #     for demo_q, demo_a in demo_tuples:
    #         prompt = _format_demo_prompt([(demo_q, demo_a)], Q_inf)
    #         completions = [icl_model.generate(prompt)[0] for _ in range(num_samples)]
    #         all_responses.append(completions)
    #     return all_responses
    
    # === FULL PARALLEL VERSION ===
    print(f"[INFO] Launching {len(demo_tuples) * num_samples} parallel inference tasks...")

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, (demo_q, demo_solution, demo_ans, j) in enumerate(demo_tuples):
            for _ in range(num_samples):
                futures.append(pool.submit(_call_generate_batch, j, demo_q, demo_solution, demo_ans, Q_inf, icl_model))

        # Collect responses
        raw_outputs = [f.result() for f in as_completed(futures)]
        # completions = [f.result() for f in futures] 
        # return completions

    # === Group by demo (demo_q used as key) ===
    from collections import defaultdict

    demo_map = defaultdict(list)
    for demo_q, out in raw_outputs:
        demo_map[demo_q].append(out)

    # Reconstruct in input order
    all_responses = []
    demo_questions = [q for q, _, _, _ in demo_tuples]
    for dq in demo_questions:
        completions = demo_map[dq]
        all_responses.append(completions)

    return all_responses
