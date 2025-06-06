# reward_aggregator.py

import re
import helpers
def extract_gold_answer(text):
    """
    Extract the final numeric answer from GSM8K-style text.
    (Gold answer still needs processing.)
    """
    if '####' in text:
        after_marker = text.split('####')[-1]
        return after_marker.strip()
    else:
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            return match.group(0)
        else:
            return text.strip().lower()

def compute_demo_accuracy(responses, gold_answer):
    """
    Args:
        responses (list[str]): model completions (clean numbers as strings)
        gold_answer (str): raw ground truth answer text (GSM8K style)

    Returns:
        accuracy (float): mean of correct predictions (0 to 1)
    """
    
    correctness = []
    for pred in responses:
        pred_answer = str(pred).strip()
        is_correct, _ = helpers.compare_latex_answers(pred_answer, gold_answer)
        correctness.append(is_correct)

    if len(correctness) == 0:
        return 0.0

    return sum(correctness) / len(correctness)
