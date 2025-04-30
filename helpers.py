# utils/helpers.py
import os
import yaml
import json
import logging
import api
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from pydantic import BaseModel, Field
import re # Added for regex parsing
from typing import List, Dict, Any, Optional, Tuple # For type hinting
# from tooldantic import ToolBaseModel, OpenAiResponseFormatGenerator

# class CustomSchemaGenerator(OpenAiResponseFormatGenerator):
#     is_inlined_refs = True
    
# class BaseModel(ToolBaseModel):
#     _schema_generator = CustomSchemaGenerator

class TextLabel(BaseModel):
    label: str
    
class NumericLabel(BaseModel):
    label: int

class MathSolution(BaseModel):
    reasoning: str = Field(..., description="Step-by-step reasoning process to solve the math problem.")
    answer: float = Field(..., description="The final numerical answer.")

class LatexSolution(BaseModel):
    reasoning: str = Field(..., description="Step-by-step reasoning process to solve the math problem.")
    answer: str = Field(..., description="The final answer in LaTeX format.")

def load_dataset_config(dataset_name):
    """Loads the configuration for a specific dataset from datasets.yaml."""
    config_path = os.path.join("utils", "datasets.yaml")
    if not os.path.exists(config_path):
        # Use basic logging/print as logger might not be set up yet
        print(f"ERROR: Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, 'r') as f:
            all_configs = yaml.safe_load(f)
        
        if not isinstance(all_configs, dict):
            raise TypeError(f"{config_path} should contain a YAML dictionary.")
            
        if dataset_name not in all_configs:
            raise KeyError(f"Dataset '{dataset_name}' not found in {config_path}")
            
        config = all_configs[dataset_name]
        # Add dataset_name to the config dict for easier reference later
        config['dataset_name'] = dataset_name
        return config
    except Exception as e:
        print(f"ERROR: Failed to load or parse config from {config_path}: {e}")
        raise
    
    
def format_examples(examples, label_mapping, input_col, target_col, logger, *, is_generation=False, verbose=False):
    """
    Formats examples for few-shot prompts, handling classification and generation.
    Args:
        examples (list): List of example dictionaries.
        label_mapping (dict | None): Mapping for classification labels.
        input_col (str): Name of the input column (e.g., 'text', 'question').
        target_col (str): Name of the target column (e.g., 'label', 'answer').
        logger: Logger instance.
        is_generation (bool): Flag indicating if it's a generation task.
        verbose (bool): Log detailed formatting info.
    Returns:
        str: Formatted examples string.
    """
    examples_text = ""
    for i, example in enumerate(examples):
        try:
            input_text = example[input_col]
            target_raw = example[target_col]

            if is_generation:
                # Default to Problem/Solution format for generation
                input_label = "Problem" # Use Problem/Solution by default
                output_label = "Solution"
                # Customize if needed based on dataset_name or config in the future
                examples_text += f"{input_label}: {input_text}\n{output_label}: {target_raw}\n\n"
            else: # Classification
                if label_mapping is None:
                     logger.error("Label mapping is required for formatting classification examples.")
                     # Skip example or raise error?
                     continue # Skip for now
                try:
                     # Map integer label to text label
                     label_text = label_mapping[target_raw] 
                     examples_text += f'Input: "{input_text}"\nLabel: {label_text}\n\n'
                except KeyError:
                     logger.warning(f"Skipping classification example {i} with invalid target key '{target_raw}' (not in label_mapping). Input: {input_text[:50]}...")
                     continue
        except KeyError as e:
            logger.warning(f"Skipping example {i} due to missing key '{e}'. Example keys: {list(example.keys())}")
            continue
        except Exception as e:
            logger.error(f"Error formatting example {i}: {e}", exc_info=True)
            continue # Skip faulty example
            
    if verbose and examples_text:
        logger.debug(f"Formatted {len(examples)} examples. Snippet:\\n{examples_text[:500]}...")
        
    return examples_text.strip() # Remove trailing newline
    
    
def parse_prediction(response, label_mapping, logger):
    """This is for datasets like TREC or SUBJ where the original labels are unsemantic integers, and has 
    mapping to semantic labels."""
    
    cleaned_response = response.strip()
    prediction = None # Initialize prediction
    # Try exact match first
    for label_id, label_text in label_mapping.items():
        if cleaned_response.upper() == label_text.upper():
            prediction = label_id
            return prediction
    
    # Fallback: Check if label text is contained (less reliable)
    if prediction is None:
        for label_id, label_text in label_mapping.items():
            if label_text.lower() in cleaned_response.lower():
                logger.warning(f"Using contained match for response '{response}' -> '{label_text}'. Exact match preferred.")
                prediction = label_id
                return prediction

    # Default if no match found
    if prediction is None:
        logger.warning(f"Unexpected response '{response}'. Could not map to any label. Assigning -1.")
        prediction = -1
    return prediction

def setup_output_dir_and_logging(*, base_results_dir, dataset_name, run_type, prompt_template, hyperparams):
    """
    Creates results directory, saves hyperparameters, sets up logging.
    Returns: tuple (logging.Logger, str) - Logger instance and results directory path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
    date = datetime.now().strftime("%Y-%m-%d")
    # Construct a clean directory name
    run_name_parts = [run_type, prompt_template]
    if "fewshot" in run_type and hyperparams.get("num_examples", 0) > 0:
        run_name_parts.append(f"{hyperparams['num_examples']}-shots")
    if "similarity" in run_type:
         # Optionally add similarity model info, keeping it concise
         sim_model_short = hyperparams.get("similarity_model", "unknown").split('/')[-1][:15] # Get last part, limit length
         run_name_parts.append(f"sim-{sim_model_short}")
    run_name_parts.append(f"seed-{hyperparams.get('random_seed', 'N')}")
    run_name_parts.append(timestamp)
    
    run_name = "_".join(run_name_parts)
    
    results_dir = os.path.join(base_results_dir, dataset_name, date, run_name)
    
    os.makedirs(results_dir, exist_ok=True)

    # Save hyperparameters
    hyperparams_path = os.path.join(results_dir, "hyperparams.yaml")
    try:
        with open(hyperparams_path, "w") as f:
            yaml.dump(hyperparams, f, default_flow_style=False)
        # Use print here as logger might be returned by this function
        print(f"Hyperparameters saved to {hyperparams_path}")
    except Exception as e:
        print(f"Error saving hyperparameters to {hyperparams_path}: {e}")

    # --- Set up logging --- 
    log_file_path = os.path.join(results_dir, "run.log")
    # Use a unique logger name to avoid conflicts if function is called multiple times
    logger_name = f"{dataset_name}_{run_name}"
    logger = logging.getLogger(logger_name)
    
    # Check if handlers already exist for this logger instance
    if not logger.handlers:
        logger.setLevel(logging.DEBUG) # Set lowest level for logger itself

        # File handler (DEBUG level)
        try:
            file_handler = logging.FileHandler(log_file_path, mode="w")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file logger at {log_file_path}: {e}")

        # Console handler (INFO level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Use a simpler formatter for console
        console_formatter = logging.Formatter('%(levelname)s: %(message)s') 
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Log initial basic info immediately
        logger.info(f"Run Log for: {logger_name}")
        logger.info(f"Results Directory: {results_dir}")
        logger.debug(f"Full Hyperparameters: {json.dumps(hyperparams)}") # Debug level for full params
    else:
        logger.info(f"Logger \'{logger_name}\' already configured.")

    return logger, results_dir

def evaluate_predictions(true_labels, predictions, label_mapping, zero_division=0):
    """
    Calculates accuracy and classification report, handling potential invalid predictions (-1).

    Args:
        true_labels (list): List of true integer labels.
        predictions (list): List of predicted integer labels (can contain -1 for failures).
        label_mapping (dict): Dictionary mapping integer labels to string names.
        zero_division (int or str): Value to return for metrics in case of zero division (default: 0).

    Returns:
        tuple: (float, dict, str) - Accuracy score, classification report dictionary, classification report string.
    """
    valid_indices = [i for i, p in enumerate(predictions) if p != -1]
    num_invalid = len(predictions) - len(valid_indices)

    true_labels_filtered = [true_labels[i] for i in valid_indices]
    predictions_filtered = [predictions[i] for i in valid_indices]

    if num_invalid > 0:
        print(f"Note: {num_invalid} predictions were invalid (-1) and excluded from metrics calculation.") # Use print or log

    if not true_labels_filtered: # Handle case where all predictions were invalid
         print("Warning: No valid predictions found to evaluate.")
         return 0.0, {}, "No valid predictions to evaluate."

    accuracy = accuracy_score(true_labels_filtered, predictions_filtered)
    expected_labels = list(label_mapping.keys()) # All potential labels
    label_names = list(label_mapping.values())  # Corresponding names

    report = classification_report(
        true_labels_filtered,
        predictions_filtered,
        labels=expected_labels, # Explicitly state all expected labels
        target_names=label_names, # Names matching expected_labels
        output_dict=True,
        zero_division=zero_division
    )
    report_text = classification_report(
        true_labels_filtered,
        predictions_filtered,
        labels=expected_labels, # Explicitly state all expected labels
        target_names=label_names, # Names matching expected_labels
        zero_division=zero_division
    )
    return accuracy, report, report_text

def save_results_json(results_dir, results_data, filename="results.json"):
    """Saves evaluation results dictionary to a JSON file."""
    results_path = os.path.join(results_dir, filename)
    try:
        with open(results_path, "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            json.dump(results_data, f, indent=2, default=convert_numpy) 
        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Error saving results to {results_path}: {e}")

def save_hard_samples(*, inputs, raw_predictions, parsed_predictions, parsed_reasonings: list, true_samples: List[Dict[str, Any]], evaluation_details: Optional[List[Dict[str, Any]]] = None, 
                      label_mapping, output_dir, dataset_name, is_generation, logger=None, 
                      target_column: Optional[str] = None, answer_column: Optional[str] = None):
    """
    Identifies misclassified/problematic samples and saves them to a CSV.
    Handles both classification and generation tasks.
    Relies on parsed_predictions and comparison with true labels/answers.
    Requires true_samples list containing the original dictionaries for generative tasks.
    Added target_column and answer_column for MATH GT extraction.
    Added parsed_reasonings list.
    """
    # If evaluation details provided, use them to identify hard samples without re-evaluating
    if evaluation_details is not None and is_generation:
        hard_samples_data = []
        for detail in evaluation_details:
            idx = detail.get("index")
            is_correct = detail.get("is_correct")
            if not is_correct:
                data = {
                    "index": idx,
                    "input": inputs[idx],
                    "raw_prediction": raw_predictions[idx],
                    "parsed_prediction": parsed_predictions[idx],
                    "parsed_reasoning": parsed_reasonings[idx] if parsed_reasonings else None,
                    "reason": ""
                }
                if dataset_name == "MATH":
                    data["true_raw_solution"] = detail.get("true_raw_solution") or true_samples[idx].get(target_column, "")
                    data["true_answer_str"] = detail.get("true_answer_str")
                    data["match_method"] = detail.get("match_method")
                    data["reason"] = f"Incorrect LaTeX answer (method: {detail.get('match_method')})"
                elif dataset_name == "gsm8k":
                    data["true_raw_answer"] = true_samples[idx].get("answer", "")
                    data["true_parsed_num"] = detail.get("true_parsed")
                    data["pred_parsed_num"] = detail.get("pred_parsed")
                    data["reason"] = "Incorrect numeric answer" if is_correct is False else "Comparison impossible"
                else:
                    data["true_raw_target"] = true_samples[idx].get(target_column if target_column else "answer", "")
                    data["reason"] = "Incorrect string answer (generic)"
                hard_samples_data.append(data)
        if hard_samples_data:
            os.makedirs(output_dir, exist_ok=True)
            hard_df = pd.DataFrame(hard_samples_data)
            hard_csv_path = os.path.join(output_dir, f"{dataset_name}_hard_samples.csv")
            try:
                hard_df.to_csv(hard_csv_path, index=False)
                if logger:
                    logger.info(f"Saved {len(hard_samples_data)} hard samples based on evaluation_results to {hard_csv_path}")
                else:
                    print(f"Saved {len(hard_samples_data)} hard samples based on evaluation_results to {hard_csv_path}")
            except Exception as e:
                if logger:
                    logger.error(f"Error saving hard samples to {hard_csv_path}: {e}")
                else:
                    print(f"Error saving hard samples to {hard_csv_path}: {e}")
        else:
            if logger:
                logger.info("No hard samples identified based on evaluation_results.")
            else:
                print("No hard samples identified based on evaluation_results.")
        return

    hard_samples_data = []
    num_parse_failures = 0 # For classification or if pred is None for generation
    num_incorrect = 0

    # Ensure lengths match for generative tasks
    if is_generation and (len(true_samples) != len(inputs) or 
                         len(parsed_predictions) != len(inputs) or 
                         len(parsed_reasonings) != len(inputs)):
        msg = f"Length mismatch in save_hard_samples: inputs({len(inputs)}), true_samples({len(true_samples)}), parsed_predictions({len(parsed_predictions)}), parsed_reasonings({len(parsed_reasonings)})"
        if logger: logger.error(msg)
        else: print(f"ERROR: {msg}")
        return # Exit if lengths don't match
        
    for i in range(len(inputs)):
        is_hard = False
        reason = ""
        pred_parsed = parsed_predictions[i]
        pred_reasoning = parsed_reasonings[i] if is_generation else None # Get reasoning for generation tasks
        true_sample = true_samples[i] if is_generation else None # Get the dict for generation

        if is_generation:
            if pred_parsed is None: # Prediction itself failed (e.g., LLM error, bad JSON)
                 is_hard = True
                 reason = "Prediction processing failed (None)"
                 num_parse_failures += 1
            else:
                # Extract ground truth based on dataset
                true_parsed_str = None
                if dataset_name == "MATH":
                    if not target_column or not answer_column:
                         if logger: logger.error("Missing target_column or answer_column for MATH in save_hard_samples call.")
                         continue # Skip sample if config missing
                    true_parsed_str = extract_math_ground_truth(true_sample, target_column, answer_column, logger)
                # Add elif for other generative datasets needing specific GT extraction
                # Example: GSM8K uses numeric comparison, not string
                elif dataset_name == "gsm8k":
                     true_parsed_num = extract_gsm8k_ground_truth(true_sample.get('answer', ''), logger)
                     pred_parsed_num = float(pred_parsed) if isinstance(pred_parsed, (int, float)) else None # Assuming pred_parsed holds the number for gsm8k
                     if true_parsed_num is not None and pred_parsed_num is not None:
                          if not np.isclose(true_parsed_num, pred_parsed_num):
                               is_hard = True
                               reason = "Incorrect numeric answer (GSM8K)"
                               num_incorrect += 1
                     elif pred_parsed_num is None:
                          is_hard = True # Treat prediction parse failure as hard
                          reason = "Prediction parsing failed (GSM8K numeric)"
                          num_parse_failures += 1
                     # No action if true_parsed_num is None (GT issue)
                     continue # Skip adding to hard samples based on GSM8K logic here, handled below
                
                else:
                    # Generic case: Assume direct string comparison if no specific logic
                    true_parsed_str = str(true_sample.get(target_column if target_column else 'answer', '')).strip()
                    pred_parsed_str = str(pred_parsed).strip()
                    if true_parsed_str and pred_parsed_str and true_parsed_str.lower() != pred_parsed_str.lower():
                         is_hard = True
                         reason = "Incorrect string answer (generic)"
                         num_incorrect += 1
                         
                # Perform comparison for MATH (and potentially others needing compare_latex_answers)
                if dataset_name == "MATH":
                    pred_parsed_str = str(pred_parsed).strip() if pred_parsed is not None else None
                    is_correct, match_method = compare_latex_answers(pred_parsed_str, true_parsed_str, logger)
                    if not is_correct:
                         is_hard = True
                         reason = f"Incorrect LaTeX answer (method: {match_method})"
                         num_incorrect += 1
                    elif is_correct is None and pred_parsed is not None: # True is None, but Pred is not
                         is_hard = True
                         reason = "Comparison impossible (True=None, Pred!=None)"
                         num_incorrect += 1 # Count as incorrect for hard sample purposes

            if is_hard:
                 # Add relevant info for generative hard samples
                 data_to_append = {
                    "index": i,
                    "input": inputs[i],
                    "raw_prediction": raw_predictions[i],
                    "parsed_prediction": pred_parsed,
                    "parsed_reasoning": pred_reasoning, # Add reasoning
                    "reason": reason # Reason for being marked hard
                 }
                 if dataset_name == "MATH":
                     data_to_append["true_raw_solution"] = true_sample.get(target_column, "")
                     data_to_append["true_answer_str"] = true_parsed_str
                 elif dataset_name == "gsm8k":
                     data_to_append["true_raw_answer"] = true_sample.get('answer', "")
                     data_to_append["true_parsed_num"] = true_parsed_num
                     data_to_append["pred_parsed_num"] = pred_parsed_num
                 else:
                     # Generic - store raw true value from target column
                     data_to_append["true_raw_target"] = true_sample.get(target_column if target_column else 'answer', "")
                 hard_samples_data.append(data_to_append)
                 
        else: # Classification
            # Hard if prediction failed parsing (-1) or was incorrect
            if pred_parsed == -1:
                 is_hard = True
                 reason = "Prediction parsing failed"
                 num_parse_failures += 1
            elif pred_parsed is not None and pred_parsed != true_sample.get(target_column if target_column else 'answer', ''):
                 is_hard = True
                 reason = "Incorrect prediction"
                 num_incorrect += 1
                 
            if is_hard:
                true_label_text = label_mapping.get(true_sample.get(target_column if target_column else 'answer', 'Unknown'), f"Unknown ({true_sample.get(target_column if target_column else 'answer', '')})")
                pred_label_text = label_mapping.get(pred_parsed, f"Invalid ({pred_parsed})") if pred_parsed != -1 else "Parse Failed"
                hard_samples_data.append({
                    "index": i,
                    "input": inputs[i],
                    "true_label_id": true_sample.get(target_column if target_column else 'answer', 'Unknown'),
                    "true_label_text": true_label_text,
                    "raw_prediction": raw_predictions[i], # Store the raw text label predicted
                    "predicted_label_id": pred_parsed,
                    "predicted_label_text": pred_label_text,
                    "reason": reason
                })

    if hard_samples_data:
        os.makedirs(output_dir, exist_ok=True)
        hard_df = pd.DataFrame(hard_samples_data)
        hard_csv_path = os.path.join(output_dir, f"{dataset_name}_hard_samples.csv")
        try:
            hard_df.to_csv(hard_csv_path, index=False)
            if logger:
                 logger.info(f"Saved {len(hard_samples_data)} hard samples ({num_parse_failures} parse failures, {num_incorrect} incorrect) to {hard_csv_path}")
            else:
                 print(f"Saved {len(hard_samples_data)} hard samples ({num_parse_failures} parse failures, {num_incorrect} incorrect) to {hard_csv_path}")
        except Exception as e:
            if logger:
                logger.error(f"Error saving hard samples to {hard_csv_path}: {e}")
            else:
                print(f"Error saving hard samples to {hard_csv_path}: {e}")
    else:
        if logger:
            logger.info("No hard samples identified or saved.")
        else:
            print("No hard samples identified or saved.")

def extract_gsm8k_ground_truth(answer_text: str, logger=None) -> float | None:
    """Extracts the final numeric answer from the GSM8K ground truth string (#### <number> pattern)."""
    # The ground truth often ends with '#### <number>'
    match = re.search(r"####\s*([\d,.]+)", answer_text)
    if match:
        try:
            # Remove commas for float conversion
            num_str = match.group(1).replace(',', '')
            return float(num_str)
        except ValueError:
            if logger:
                logger.error(f"Could not convert extracted GSM8K ground truth answer '{match.group(1)}' to float.")
            return None
    # Attempt backup: Look for the last number in the string if primary pattern fails
    numbers = re.findall(r'[-+]?(\d+,)*\d+(\.\d+)?', answer_text)
    if numbers:
        try:
            last_num_str = numbers[-1][0].replace(',','') # Get the last number found
            if logger:
                logger.warning(f"Found pattern '#### ...' failed for GT: '{answer_text}'. Using last number found: '{last_num_str}'")
            return float(last_num_str)
        except ValueError:
             if logger:
                 logger.error(f"Could not convert backup extracted GSM8K ground truth answer '{last_num_str}' to float.")
             return None
             
    if logger:
        logger.warning(f"Could not find numeric answer pattern in GSM8K ground truth: {answer_text}")
    return None

def extract_math_ground_truth(sample: Dict[str, Any], target_column: str, answer_column: Optional[str], logger: Optional[logging.Logger] = None) -> str | None:
    """Extracts the ground truth answer string for the MATH dataset.
    
    Prioritizes the dedicated 'answer_column' if provided and present in the sample.
    Otherwise, falls back to the 'target_column' (e.g., 'solution').
    Returns the raw string value.
    """
    if answer_column and answer_column in sample:
        true_answer_raw = sample[answer_column]
        if isinstance(true_answer_raw, str):
            return true_answer_raw.strip()
        else:
            # Attempt to convert non-string answers if they exist
            try:
                return str(true_answer_raw).strip()
            except Exception as e:
                if logger:
                    logger.warning(f"Could not convert ground truth from answer_column '{answer_column}' to string for sample: {sample}. Error: {e}")
                return None
    elif target_column in sample:
        true_answer_raw = sample[target_column]
        if isinstance(true_answer_raw, str):
             # If falling back to solution, still try to extract \boxed{} first
             match = re.search(r"\\boxed{(.*?)}", true_answer_raw)
             if match:
                 if logger:
                     logger.debug(f"Extracted answer from \boxed{{}} in target column '{target_column}'.")
                 return match.group(1).strip()
             else:
                 # If no \boxed{}, return the whole raw solution string from target_column
                 return true_answer_raw.strip()
        else:
            try:
                return str(true_answer_raw).strip()
            except Exception as e:
                 if logger:
                     logger.warning(f"Could not convert ground truth from target_column '{target_column}' to string for sample: {sample}. Error: {e}")
                 return None
    else:
        if logger:
            logger.error(f"Neither answer_column '{answer_column}' nor target_column '{target_column}' found in MATH sample: {sample}")
        return None
def compare_latex_answers(pred_ans: Optional[str], true_ans: Optional[str], logger: Optional[logging.Logger] = None, context: Optional[str] = None, use_llm: bool = True) -> Tuple[bool | None, str]:
    """Compares predicted LaTeX answer string with the ground truth string.

    1. Checks for None inputs.
    2. Performs case-insensitive exact string comparison.
    3. If #2 fails, performs digit sequence comparison.
    4. For text answers, compares after removing LaTeX formatting.

    Returns:
        Tuple[bool | None, str]: (Comparison result, reason/method)
            - Comparison result: True if match, False if mismatch, None if comparison impossible.
            - Reason: 'exact_match', 'digit_match', 'text_match', 'mismatch', 'none_input'.
    """
    if pred_ans is None or true_ans is None:
        return None, "none_input" # Cannot compare if either is missing

    # Clean the LaTeX strings: remove spaces, \left, and \right
    pred_clean = pred_ans.strip().replace(" ", "").replace("\\left", "").replace("\\right", "").replace("$", "")
    true_clean = true_ans.strip().replace(" ", "").replace("\\left", "").replace("\\right", "").replace("$", "")

    # Remove \text{} and \boxed{} constructs
    pred_clean = re.sub(r"\\text\{(.*?)\}", r"\1", pred_clean)
    pred_clean = re.sub(r"\\boxed\{(.*?)\}", r"\1", pred_clean)
    true_clean = re.sub(r"\\text\{(.*?)\}", r"\1", true_clean)
    true_clean = re.sub(r"\\boxed\{(.*?)\}", r"\1", true_clean)

    # Attempt numeric match
    try:
        # print(f"pred_clean: {pred_clean}, true_clean: {true_clean}")
        if float(pred_ans) == float(true_ans):
            return True, "numeric_match"
    except Exception:
        pass  # fallback to other comparisons
    
    
    # 1. Case-insensitive exact match
    if pred_clean.lower() == true_clean.lower():
        return True, "exact_match"

    # 2. Text-only comparison (remove LaTeX formatting and compare alphabetic characters)
    # Only apply this if both answers are completely text (no numbers at all)
    # Check if the answers contain any digits
    # Only proceed if both answers consist *entirely* of letters
    pred_is_alpha = all(c.isalpha() for c in pred_clean)
    true_is_alpha = all(c.isalpha() for c in true_clean)

    if pred_is_alpha and true_is_alpha:
        # Then remove all LaTeX‐related characters and keep only alphabetic chars
        pred_alpha = "".join(c for c in pred_clean if c.isalpha()).lower()
        true_alpha = "".join(c for c in true_clean if c.isalpha()).lower()
        
        if pred_alpha and true_alpha and pred_alpha == true_alpha:
            message = f"Using text-only match after removing LaTeX formatting: '{pred_alpha}'"
            if logger:
                logger.info(message)
            return True, "text_match"
        
    # 3. Check for 10 == 10.0 by stripping all non-numeric characters except decimal point
    pred_digits = re.sub(r'[^0-9. ]', '', pred_clean)
    true_digits = re.sub(r'[^0-9. ]', '', true_clean)
    try:
        if  pred_digits != "":
            if pred_digits == true_digits:
                return True, "integer_float_match"
            if float(pred_digits) == float(true_digits):
                return True, "integer_float_match"
            elif pred_digits == pred_clean and true_digits == true_clean:
                return pred_clean == true_clean or float(pred_clean) == float(true_clean), "integer_float_match"
    except Exception:
        pass    
    # 4. Check if both prediction and true value consist *only* of a latex fraction
    # Note: Double backslashes are needed for Python strings, then doubled again for regex literal backslash
    latex_frac_pattern = r"\\\\frac\{.*?\}\{.*?\}" # Corrected: Removed trailing backslash
    pred_is_frac_only = re.fullmatch(latex_frac_pattern, pred_clean)
    true_is_frac_only = re.fullmatch(latex_frac_pattern, true_clean)

    if pred_is_frac_only and true_is_frac_only:
        # If both strings are *exactly* in the format \\frac{...}{...}, compare them directly
        is_match = pred_clean == true_clean
        match_type = "exact_latex_fraction_match" if is_match else "mismatched_latex_fraction"
        return is_match, match_type

    # 5. Fallback to simplified exact match (add other checks above this)
    # if = sign is present, split it by the rightmost = sign and compare the two parts
    pred_clean_equals = pred_clean
    true_clean_equals = true_clean
    if "=" in pred_clean:
        pred_clean_equals = pred_clean.split("=")[-1].strip()
    if "=" in true_clean:
        true_clean_equals = true_clean.split("=")[-1].strip()
    if pred_clean_equals == true_clean_equals:
        return True, "simplify_equals_match"
    
    # 3. Check for 10 == 10.0 by stripping all non-numeric characters except decimal point after splitting by = sign
    pred_digits = re.sub(r'[^0-9.]', '', pred_clean_equals)
    true_digits = re.sub(r'[^0-9.]', '', true_clean_equals)
    
    if  pred_digits != "":
        if pred_digits == true_digits:
            return True, "integer_float_match"
        elif pred_digits == pred_clean and true_digits == true_clean:
            return pred_clean == true_clean, "integer_float_match"
    
    # 4. Use LLM to compare the solutions
    if use_llm:
        prompt = {}
        prompt['system'] = f"""You are a math expert, and you're annotating whether or not a mathametical solution can be interpreted as the correct answer for the problem: {context}"""
        prompt['user'] = f"""Attempted Solution: "{pred_ans}"\n\nTrue Solution: "{true_ans}"\n\nReturn 'True' if the attempted solution can be interpreted as correct in the context of the problem, 'False' if it can't."""
        # print(prompt)
        response,_ = api.query_llm(
            prompt=prompt,
            model="gpt-4.1-nano",
            debug=False,
            system_prompt_included=True  
        )
        # print(response)
        if len(response) < 10:
            return response.replace("'", "").replace('"', "").replace(".", "").lower() == "true", "llm_comparison"
        else:
            return 'true' in response.split("\n")[-1].lower(), "llm_comparison"
        
    # Check for = sign and take everything after it as the answer
    pred_clean = re.sub(r"=.*", "", pred_clean)
    true_clean = re.sub(r"=.*", "", true_clean)
    
    # 4. Digit sequence comparison (for numeric answers)
    pred_digits = "".join(filter(str.isdigit, pred_clean))
    true_digits = "".join(filter(str.isdigit, true_clean))

    if pred_digits == true_digits and pred_digits != "": # Ensure not comparing empty strings
        message = f"Exact match failed for pred='{pred_clean}', true='{true_clean}'. Using digit sequence match ('{pred_digits}')."
        if logger:
            logger.warning(message)
        else:
            print(f"WARNING: {message}")
        return True, "digit_match"
    
    return False, "mismatch"

def evaluate_gsm8k_predictions(true_answers_raw: list, parsed_predictions: list, logger) -> dict:
    """
    Evaluates GSM8K predictions by comparing parsed numeric answers.
    Assumes parsed_predictions contains floats or None (from MathSolution.answer).
    """
    correct_count = 0
    valid_comparisons = 0
    results = {'details': []}

    for i, (true_raw, pred_parsed) in enumerate(zip(true_answers_raw, parsed_predictions)):
        # Extract numeric ground truth
        true_parsed = extract_gsm8k_ground_truth(true_raw, logger)
        is_correct = None # Use None to indicate comparison wasn't possible

        if true_parsed is not None and pred_parsed is not None:
            # Use np.isclose for robust float comparison
            if np.isclose(true_parsed, pred_parsed):
                correct_count += 1
                is_correct = True
            else:
                is_correct = False
            valid_comparisons += 1
        elif true_parsed is None:
             # Logged within extract_gsm8k_ground_truth
             pass
        elif pred_parsed is None:
             # This indicates the LLM failed to return valid JSON or the answer field was missing/invalid
             logger.warning(f"Prediction for sample {i} was None (JSON parsing failed or invalid answer field?).")

        results['details'].append({
            'index': i,
            'true_raw': true_raw,
            'true_parsed': true_parsed,
            'pred_parsed': pred_parsed, # This is the float or None from MathSolution
            'is_correct': is_correct
        })

    accuracy = (correct_count / valid_comparisons) if valid_comparisons > 0 else 0.0
    logger.info(f"GSM8K Evaluation: Correct={correct_count}, Valid Comparisons={valid_comparisons}, Accuracy={accuracy:.4f}")
    results['accuracy'] = accuracy
    results['correct_count'] = correct_count
    results['valid_comparisons'] = valid_comparisons
    results['total_samples'] = len(true_answers_raw)
    return results

def evaluate_math_predictions(true_samples: List[Dict[str, Any]], parsed_predictions: list, parsed_reasonings: list, target_column: str, answer_column: Optional[str], logger) -> dict:
    """
    Evaluates MATH predictions by comparing predicted LaTeX answer strings.
    Assumes parsed_predictions contains strings or None (from LatexSolution.answer).
    Assumes parsed_reasonings contains strings or None.
    Uses extract_math_ground_truth and compare_latex_answers.
    """
    correct_count = 0
    compared_count = 0 # Count comparisons that were possible
    match_types = {'exact_match': 0, 'digit_match': 0, 'mismatch': 0, 'text_match':0, 'integer_float_match':0, 'simplify_equals_match':0, 'none_input': 0, 'llm_comparison': 0}
    results = {'details': []}

    # Ensure lengths match
    if len(true_samples) != len(parsed_predictions) or len(true_samples) != len(parsed_reasonings):
         logger.error(f"Length mismatch in evaluate_math_predictions: true_samples({len(true_samples)}), parsed_predictions({len(parsed_predictions)}), parsed_reasonings({len(parsed_reasonings)}). Returning partial results.")
         # Optionally handle this differently, e.g., raise error or truncate
         min_len = min(len(true_samples), len(parsed_predictions), len(parsed_reasonings))
         true_samples = true_samples[:min_len]
         parsed_predictions = parsed_predictions[:min_len]
         parsed_reasonings = parsed_reasonings[:min_len]

    for i, (true_sample, pred_parsed_ans) in enumerate(zip(true_samples, parsed_predictions)):
        # Extract ground truth string using the MATH-specific function
        true_parsed_str = extract_math_ground_truth(true_sample, target_column, answer_column, logger)
        
        # Get predicted answer string and reasoning
        pred_parsed_str = str(pred_parsed_ans).strip() if pred_parsed_ans is not None else None
        pred_reasoning = parsed_reasonings[i] # Get reasoning from the corresponding list
             
        # Compare using the new function
        is_correct, match_method = compare_latex_answers(pred_parsed_str, true_parsed_str, context=true_sample.get('problem', '[Input Missing]'), logger=logger)
        
        match_types[match_method] += 1
        
        if is_correct is not None: # If comparison was possible
            compared_count += 1
            if is_correct:
                correct_count += 1
        # else: comparison was impossible (None input)

        results['details'].append({
            'index': i,
            'input': true_sample.get('problem', '[Input Missing]'), # Assuming input column is 'problem'
            'true_raw_solution': true_parsed_str,
            'true_answer_str': true_parsed_str,
            'pred_latex_str': pred_parsed_str,
            'pred_reasoning': pred_reasoning, # Added reasoning
            'is_correct': is_correct,
            'match_method': match_method
        })

    accuracy = (correct_count / compared_count) if compared_count > 0 else 0.0
    logger.info(f"MATH Evaluation: Correct={correct_count}, Compared={compared_count}, Accuracy={accuracy:.4f}")
    logger.info(f"Match breakdown: {match_types}")
    results['accuracy'] = accuracy
    results['correct_count'] = correct_count
    results['compared_count'] = compared_count
    results['total_samples'] = len(true_samples)
    results['match_types'] = match_types
    return results

def load_jsonl(file_path: str, logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """Loads data from a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.
        logger (Optional[logging.Logger]): Logger instance for logging errors.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a line in the JSONL file.
                                Returns an empty list if the file is not found or cannot be parsed.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        message = f"Error decoding JSON on line {line_num + 1} in {file_path}: {e}"
                        if logger:
                            logger.error(message)
                        else:
                            print(f"ERROR: {message}")
                        # Decide whether to skip the line or raise the error
                        # For robustness, let's skip the line and continue
                        continue 
    except FileNotFoundError:
        message = f"File not found: {file_path}"
        if logger:
            logger.error(message)
        else:
            print(f"ERROR: {message}")
        return [] # Return empty list if file not found
    except Exception as e:
        message = f"An unexpected error occurred while reading {file_path}: {e}"
        if logger:
            logger.error(message, exc_info=True)
        else:
            print(f"ERROR: {message}")
        return [] # Return empty list on other errors

    return data
