import torch
import os

import pandas as pd
import random
from collections import defaultdict

import math
from tqdm import tqdm
import random
import itertools

# import pysvelte
import math
from transformer_lens import HookedTransformer
from typing import Literal, Tuple


def add_space(x):
    if x[0] != " ":
        return " " + x
    return x


def create_wino_bias_dataset(model_name: str,  save_dir: str, seed: int = 42) -> None:
    """
    Generate WinoBias dataset.

    This function creates datasets of profession-based sentences following the WinoBias format,
    with different combinations of male/female professions and pronouns. It generates:
    - Pro/anti stereotypical examples

    Args:
        model_name (str): Name of the model being evaluated

    Returns:
        None: Saves generated datasets as CSV files in data/{model_name}/wino_bias/{seed}/
        with filenames indicating the configuration:
        - pro/anti: Whether example follows or contradicts stereotypes
        - male/female: Gender being referenced
        - first/second_prof: Position of the profession being referred to

    Files contain columns:
        - prompt: The full text prompt
        - id: Unique identifier
        - pair_id: ID grouping related examples
        - correct_profession_idx: Index of correct profession (1/2)
        - correct/wrong_token: The profession tokens
        - Various position indices for tokens in the sequence

    """

    MALE_PROF_llama3 = [
        "lawyer",
        "farmer",
        "physician",
        "developer",
        "manager",
        "guard",
        "mechanic",
        "chief",
        "driver",
        "mover",
        "supervisor",
        "analyst",
        "constructor",
        "cook",
        "sheriff"]

    FEMALE_PROF_llama3 = [

        "teacher",
        "editor",
        "cashier",
        "writer",
        "counselor",
        "counselors",
        "nurse",
        "sewer",
        "baker",
        "auditor",
        "designer",
        "clerk",
        "cleaner",
        "librarian",
        "secretary",
        "assistant",
        "attendant",
        "accountant"
    ]

    MALE_PROF_gp2 = [
        "sheriff", "chief", "driver", "mechanic", "lawyer",
        "cook", "guard", "farmer", "supervisor", "analyst",
        "constructor", "physician", "developer", "manager"
    ]
    FEMALE_PROF_gtp2 = [
        "clerk", "baker", "attendant", "auditor", "nurse",
        "accountant", "writer", "editor", "teacher", "assistant",
        "cleaner", "secretary", "counselor", "counselors",
        "sewer", "designer"
    ]

    FEMALE_PROF = FEMALE_PROF_llama3 if "Llama" in model_name else FEMALE_PROF_gtp2
    MALE_PROF = MALE_PROF_llama3 if "Llama" in model_name else MALE_PROF_gp2
    dtype = "bf16" if "Llama" else "float32"

    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    model_name = model_name.split("/")[-1]
    data_dir = "data"

    templates = pd.read_csv("data_utils/templates.csv")
    templates = templates[templates["answer"] == 2]
    random.seed(seed)
    split = random.choices(["circuit", "eval", "ablation",], weights=[
                           45, 45, 20], k=len(FEMALE_PROF) * len(MALE_PROF) * templates.shape[0])
    dataset = defaultdict(list)
    for index, row in templates.iterrows():
        for f in FEMALE_PROF:
            for m in MALE_PROF:

                first_profession = f
                second_profession = m
                prompt = row["prompt"].format(
                    first_profession, second_profession, row["female_pronoun"], row["female_pronoun"])
                tokens_list = model.to_str_tokens(prompt, prepend_bos=True)

                first_profession_tokenized = model.to_str_tokens(
                    add_space(first_profession), prepend_bos=False)
                second_profession_tokenized = model.to_str_tokens(
                    add_space(second_profession), prepend_bos=False)

                first_profession_index = tokens_list.index(
                    first_profession_tokenized[0])
                second_profession_index = tokens_list.index(
                    second_profession_tokenized[0])

                first_pronoun = tokens_list.index(
                    add_space(row["female_pronoun"]))

                dataset["prompt"].append(prompt)
                id = 2 * row["id"] if row["answer"] == 1 else 2 * row["id"] + 1
                dataset["id"].append(id)
                dataset["pair_id"].append(row["id"])
                dataset["correct_proffesion_idx"].append(row["answer"])
                dataset["correct_token"].append(second_profession_tokenized[0])
                dataset["wrong_token"].append(first_profession_tokenized[0])
                dataset["wrong_profession"].append(1)
                dataset["interaction"].append(
                    first_profession_index + len(first_profession_tokenized))
                dataset["correct_profession"].append(
                    second_profession_index - 1)
                dataset["conjunction"].append(
                    second_profession_index + len(second_profession_tokenized))
                dataset["first_pronoun"].append(first_pronoun)
                dataset["circumstances"].append(first_pronoun + 1)
                dataset["dot"].append(tokens_list.index("."))
                dataset["The"].append(tokens_list.index(".") + 1)
                dataset["pronoun"].append(tokens_list.index(".") + 2)
                dataset["second_pronoun"].append(tokens_list.index(
                    ".") + 4 if "Llama" in model_name else tokens_list.index(".") + 3)
                dataset["refers"].append(tokens_list.index(
                    ".") + 5 if "Llama" in model_name else tokens_list.index(".") + 4)
                dataset["to"].append(tokens_list.index(
                    ".") + 6 if "Llama" in model_name else tokens_list.index(".") + 5)
                dataset["the"].append(tokens_list.index(
                    ".") + 7 if "Llama" in model_name else tokens_list.index(".") + 6)
                dataset["length"].append(len(tokens_list))

    dataset["split"] = split
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    path = os.path.join(save_dir, "anti_female_second_prof.csv")
    print(path)
    dataset.to_csv(path)

    dataset = defaultdict(list)
    for index, row in templates.iterrows():
        for f in FEMALE_PROF:
            for m in MALE_PROF:

                first_profession = f
                second_profession = m
                prompt = row["prompt"].format(
                    first_profession, second_profession, row["male_pronoun"], row["male_pronoun"])
                tokens_list = model.to_str_tokens(prompt, prepend_bos=True)

                first_profession_tokenized = model.to_str_tokens(
                    add_space(first_profession), prepend_bos=False)
                second_profession_tokenized = model.to_str_tokens(
                    add_space(second_profession), prepend_bos=False)

                first_profession_index = tokens_list.index(
                    first_profession_tokenized[0])
                second_profession_index = tokens_list.index(
                    second_profession_tokenized[0])

                first_pronoun = tokens_list.index(
                    add_space(row["male_pronoun"]))

                dataset["prompt"].append(prompt)
                id = 2 * row["id"] if row["answer"] == 1 else 2 * row["id"] + 1
                dataset["id"].append(id)
                dataset["pair_id"].append(row["id"])
                dataset["correct_proffesion_idx"].append(row["answer"])
                dataset["correct_token"].append(second_profession_tokenized[0])
                dataset["wrong_token"].append(first_profession_tokenized[0])
                dataset["wrong_profession"].append(1)
                dataset["interaction"].append(
                    first_profession_index + len(first_profession_tokenized))
                dataset["correct_profession"].append(
                    second_profession_index - 1)
                dataset["conjunction"].append(
                    second_profession_index + len(second_profession_tokenized))
                dataset["first_pronoun"].append(first_pronoun)
                dataset["circumstances"].append(first_pronoun + 1)
                dataset["dot"].append(tokens_list.index("."))
                dataset["The"].append(tokens_list.index(".") + 1)
                dataset["pronoun"].append(tokens_list.index(".") + 2)
                dataset["second_pronoun"].append(tokens_list.index(
                    ".") + 4 if "Llama" in model_name else tokens_list.index(".") + 3)
                dataset["refers"].append(tokens_list.index(
                    ".") + 5 if "Llama" in model_name else tokens_list.index(".") + 4)
                dataset["to"].append(tokens_list.index(
                    ".") + 6 if "Llama" in model_name else tokens_list.index(".") + 5)
                dataset["the"].append(tokens_list.index(
                    ".") + 7 if "Llama" in model_name else tokens_list.index(".") + 6)
                dataset["length"].append(len(tokens_list))

    dataset["split"] = split
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    path = os.path.join(save_dir, "pro_male_second_prof.csv")
    dataset.to_csv(path)

    dataset = defaultdict(list)
    for index, row in templates.iterrows():
        for f in FEMALE_PROF:
            for m in MALE_PROF:

                first_profession = m
                second_profession = f
                prompt = row["prompt"].format(
                    first_profession, second_profession, row["female_pronoun"], row["female_pronoun"])
                tokens_list = model.to_str_tokens(prompt, prepend_bos=True)

                first_profession_tokenized = model.to_str_tokens(
                    add_space(first_profession), prepend_bos=False)
                second_profession_tokenized = model.to_str_tokens(
                    add_space(second_profession), prepend_bos=False)

                first_profession_index = tokens_list.index(
                    first_profession_tokenized[0])
                second_profession_index = tokens_list.index(
                    second_profession_tokenized[0])

                first_pronoun = tokens_list.index(
                    add_space(row["female_pronoun"]))

                dataset["prompt"].append(prompt)
                id = 2 * row["id"] if row["answer"] == 1 else 2 * row["id"] + 1
                dataset["id"].append(id)
                dataset["pair_id"].append(row["id"])
                dataset["correct_proffesion_idx"].append(row["answer"])
                dataset["correct_token"].append(second_profession_tokenized[0])
                dataset["wrong_token"].append(first_profession_tokenized[0])
                dataset["wrong_profession"].append(1)
                dataset["interaction"].append(
                    first_profession_index + len(first_profession_tokenized))
                dataset["correct_profession"].append(
                    second_profession_index - 1)
                dataset["conjunction"].append(
                    second_profession_index + len(second_profession_tokenized))
                dataset["first_pronoun"].append(first_pronoun)
                dataset["circumstances"].append(first_pronoun + 1)
                dataset["dot"].append(tokens_list.index("."))
                dataset["The"].append(tokens_list.index(".") + 1)
                dataset["pronoun"].append(tokens_list.index(".") + 2)
                dataset["second_pronoun"].append(tokens_list.index(
                    ".") + 4 if "Llama" in model_name else tokens_list.index(".") + 3)
                dataset["refers"].append(tokens_list.index(
                    ".") + 5 if "Llama" in model_name else tokens_list.index(".") + 4)
                dataset["to"].append(tokens_list.index(
                    ".") + 6 if "Llama" in model_name else tokens_list.index(".") + 5)
                dataset["the"].append(tokens_list.index(
                    ".") + 7 if "Llama" in model_name else tokens_list.index(".") + 6)
                dataset["length"].append(len(tokens_list))

    dataset["split"] = split
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    path = os.path.join(save_dir, "pro_female_second_prof.csv")
    dataset.to_csv(path)

    dataset = defaultdict(list)
    for index, row in templates.iterrows():
        for f in FEMALE_PROF:
            for m in MALE_PROF:

                first_profession = m
                second_profession = f
                prompt = row["prompt"].format(
                    first_profession, second_profession, row["male_pronoun"], row["male_pronoun"])
                tokens_list = model.to_str_tokens(prompt, prepend_bos=True)

                first_profession_tokenized = model.to_str_tokens(
                    add_space(first_profession), prepend_bos=False)
                second_profession_tokenized = model.to_str_tokens(
                    add_space(second_profession), prepend_bos=False)

                first_profession_index = tokens_list.index(
                    first_profession_tokenized[0])
                second_profession_index = tokens_list.index(
                    second_profession_tokenized[0])

                first_pronoun = tokens_list.index(
                    add_space(row["male_pronoun"]))

                dataset["prompt"].append(prompt)
                id = 2 * row["id"] if row["answer"] == 1 else 2 * row["id"] + 1
                dataset["id"].append(id)
                dataset["pair_id"].append(row["id"])
                dataset["correct_proffesion_idx"].append(row["answer"])
                dataset["correct_token"].append(second_profession_tokenized[0])
                dataset["wrong_token"].append(first_profession_tokenized[0])
                dataset["wrong_profession"].append(1)
                dataset["interaction"].append(
                    first_profession_index + len(first_profession_tokenized))
                dataset["correct_profession"].append(
                    second_profession_index - 1)
                dataset["conjunction"].append(
                    second_profession_index + len(second_profession_tokenized))
                dataset["first_pronoun"].append(first_pronoun)
                dataset["circumstances"].append(first_pronoun + 1)
                dataset["dot"].append(tokens_list.index("."))
                dataset["The"].append(tokens_list.index(".") + 1)
                dataset["pronoun"].append(tokens_list.index(".") + 2)
                dataset["second_pronoun"].append(tokens_list.index(
                    ".") + 4 if "Llama" in model_name else tokens_list.index(".") + 3)
                dataset["refers"].append(tokens_list.index(
                    ".") + 5 if "Llama" in model_name else tokens_list.index(".") + 4)
                dataset["to"].append(tokens_list.index(
                    ".") + 6 if "Llama" in model_name else tokens_list.index(".") + 5)
                dataset["the"].append(tokens_list.index(
                    ".") + 7 if "Llama" in model_name else tokens_list.index(".") + 6)
                dataset["length"].append(len(tokens_list))

    dataset["split"] = split
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    path = os.path.join(save_dir, "anti_male_second_prof.csv")
    dataset.to_csv(path)

    #######################################

    templates = pd.read_csv("data_utils/templates.csv")
    templates = templates[templates["answer"] == 1]

    dataset = defaultdict(list)
    for index, row in templates.iterrows():
        for f in FEMALE_PROF:
            for m in MALE_PROF:

                first_profession = m
                second_profession = f
                prompt = row["prompt"].format(
                    first_profession, second_profession, row["female_pronoun"], row["female_pronoun"])
                tokens_list = model.to_str_tokens(prompt, prepend_bos=True)

                first_profession_tokenized = model.to_str_tokens(
                    add_space(first_profession), prepend_bos=False)
                second_profession_tokenized = model.to_str_tokens(
                    add_space(second_profession), prepend_bos=False)

                first_profession_index = tokens_list.index(
                    first_profession_tokenized[0])
                second_profession_index = tokens_list.index(
                    second_profession_tokenized[0])

                first_pronoun = tokens_list.index(
                    add_space(row["female_pronoun"]))

                dataset["prompt"].append(prompt)
                id = 2 * row["id"] if row["answer"] == 1 else 2 * row["id"] + 1
                dataset["id"].append(id)
                dataset["pair_id"].append(row["id"])
                dataset["correct_proffesion_idx"].append(row["answer"])
                dataset["correct_token"].append(first_profession_tokenized[0])
                dataset["wrong_token"].append(second_profession_tokenized[0])
                dataset["correct_profession"].append(1)
                dataset["interaction"].append(
                    first_profession_index + len(first_profession_tokenized))
                dataset["wrong_profession"].append(second_profession_index - 1)
                dataset["conjunction"].append(
                    second_profession_index + len(second_profession_tokenized))
                dataset["first_pronoun"].append(first_pronoun)
                dataset["circumstances"].append(first_pronoun + 1)
                dataset["dot"].append(tokens_list.index("."))
                dataset["The"].append(tokens_list.index(".") + 1)
                dataset["pronoun"].append(tokens_list.index(".") + 2)
                dataset["second_pronoun"].append(tokens_list.index(
                    ".") + 4 if "Llama" in model_name else tokens_list.index(".") + 3)
                dataset["refers"].append(tokens_list.index(
                    ".") + 5 if "Llama" in model_name else tokens_list.index(".") + 4)
                dataset["to"].append(tokens_list.index(
                    ".") + 6 if "Llama" in model_name else tokens_list.index(".") + 5)
                dataset["the"].append(tokens_list.index(
                    ".") + 7 if "Llama" in model_name else tokens_list.index(".") + 6)
                dataset["length"].append(len(tokens_list))

    dataset["split"] = split
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    path = os.path.join(save_dir, "anti_female_first_prof.csv")
    dataset.to_csv(path)

    dataset = defaultdict(list)
    for index, row in templates.iterrows():
        for f in FEMALE_PROF:
            for m in MALE_PROF:

                first_profession = m
                second_profession = f
                prompt = row["prompt"].format(
                    first_profession, second_profession, row["male_pronoun"], row["male_pronoun"])
                tokens_list = model.to_str_tokens(prompt, prepend_bos=True)

                first_profession_tokenized = model.to_str_tokens(
                    add_space(first_profession), prepend_bos=False)
                second_profession_tokenized = model.to_str_tokens(
                    add_space(second_profession), prepend_bos=False)

                first_profession_index = tokens_list.index(
                    first_profession_tokenized[0])
                second_profession_index = tokens_list.index(
                    second_profession_tokenized[0])

                first_pronoun = tokens_list.index(
                    add_space(row["male_pronoun"]))

                dataset["prompt"].append(prompt)
                id = 2 * row["id"] if row["answer"] == 1 else 2 * row["id"] + 1
                dataset["id"].append(id)
                dataset["pair_id"].append(row["id"])
                dataset["correct_proffesion_idx"].append(row["answer"])
                dataset["correct_token"].append(first_profession_tokenized[0])
                dataset["wrong_token"].append(second_profession_tokenized[0])
                dataset["correct_profession"].append(1)
                dataset["interaction"].append(
                    first_profession_index + len(first_profession_tokenized))
                dataset["wrong_profession"].append(second_profession_index - 1)
                dataset["conjunction"].append(
                    second_profession_index + len(second_profession_tokenized))
                dataset["first_pronoun"].append(first_pronoun)
                dataset["circumstances"].append(first_pronoun + 1)
                dataset["dot"].append(tokens_list.index("."))
                dataset["The"].append(tokens_list.index(".") + 1)
                dataset["pronoun"].append(tokens_list.index(".") + 2)
                dataset["second_pronoun"].append(tokens_list.index(
                    ".") + 4 if "Llama" in model_name else tokens_list.index(".") + 3)
                dataset["refers"].append(tokens_list.index(
                    ".") + 5 if "Llama" in model_name else tokens_list.index(".") + 4)
                dataset["to"].append(tokens_list.index(
                    ".") + 6 if "Llama" in model_name else tokens_list.index(".") + 5)
                dataset["the"].append(tokens_list.index(
                    ".") + 7 if "Llama" in model_name else tokens_list.index(".") + 6)
                dataset["length"].append(len(tokens_list))

    dataset["split"] = split
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    path = os.path.join(save_dir, "pro_male_first_prof.csv")
    dataset.to_csv(path)

    dataset = defaultdict(list)
    for index, row in templates.iterrows():
        for f in FEMALE_PROF:
            for m in MALE_PROF:

                first_profession = f
                second_profession = m
                prompt = row["prompt"].format(
                    first_profession, second_profession, row["female_pronoun"], row["female_pronoun"])
                tokens_list = model.to_str_tokens(prompt, prepend_bos=True)

                first_profession_tokenized = model.to_str_tokens(
                    add_space(first_profession), prepend_bos=False)
                second_profession_tokenized = model.to_str_tokens(
                    add_space(second_profession), prepend_bos=False)

                first_profession_index = tokens_list.index(
                    first_profession_tokenized[0])
                second_profession_index = tokens_list.index(
                    second_profession_tokenized[0])

                first_pronoun = tokens_list.index(
                    add_space(row["female_pronoun"]))

                dataset["prompt"].append(prompt)
                id = 2 * row["id"] if row["answer"] == 1 else 2 * row["id"] + 1
                dataset["id"].append(id)
                dataset["pair_id"].append(row["id"])
                dataset["correct_proffesion_idx"].append(row["answer"])
                dataset["correct_token"].append(first_profession_tokenized[0])
                dataset["wrong_token"].append(second_profession_tokenized[0])
                dataset["correct_profession"].append(1)
                dataset["interaction"].append(
                    first_profession_index + len(first_profession_tokenized))
                dataset["wrong_profession"].append(second_profession_index - 1)
                dataset["conjunction"].append(
                    second_profession_index + len(second_profession_tokenized))
                dataset["first_pronoun"].append(first_pronoun)
                dataset["circumstances"].append(first_pronoun + 1)
                dataset["dot"].append(tokens_list.index("."))
                dataset["The"].append(tokens_list.index(".") + 1)
                dataset["pronoun"].append(tokens_list.index(".") + 2)
                dataset["second_pronoun"].append(tokens_list.index(
                    ".") + 4 if "Llama" in model_name else tokens_list.index(".") + 3)
                dataset["refers"].append(tokens_list.index(
                    ".") + 5 if "Llama" in model_name else tokens_list.index(".") + 4)
                dataset["to"].append(tokens_list.index(
                    ".") + 6 if "Llama" in model_name else tokens_list.index(".") + 5)
                dataset["the"].append(tokens_list.index(
                    ".") + 7 if "Llama" in model_name else tokens_list.index(".") + 6)
                dataset["length"].append(len(tokens_list))

    dataset["split"] = split
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    path = os.path.join(save_dir, "pro_female_first_prof.csv")
    dataset.to_csv(path)

    dataset = defaultdict(list)
    for index, row in templates.iterrows():
        for f in FEMALE_PROF:
            for m in MALE_PROF:

                first_profession = f
                second_profession = m
                prompt = row["prompt"].format(
                    first_profession, second_profession, row["male_pronoun"], row["male_pronoun"])
                tokens_list = model.to_str_tokens(prompt, prepend_bos=True)

                first_profession_tokenized = model.to_str_tokens(
                    add_space(first_profession), prepend_bos=False)
                second_profession_tokenized = model.to_str_tokens(
                    add_space(second_profession), prepend_bos=False)

                first_profession_index = tokens_list.index(
                    first_profession_tokenized[0])
                second_profession_index = tokens_list.index(
                    second_profession_tokenized[0])

                first_pronoun = tokens_list.index(
                    add_space(row["male_pronoun"]))

                dataset["prompt"].append(prompt)
                id = 2 * row["id"] if row["answer"] == 1 else 2 * row["id"] + 1
                dataset["id"].append(id)
                dataset["pair_id"].append(row["id"])
                dataset["correct_proffesion_idx"].append(row["answer"])
                dataset["correct_token"].append(first_profession_tokenized[0])
                dataset["wrong_token"].append(second_profession_tokenized[0])
                dataset["correct_profession"].append(1)
                dataset["interaction"].append(
                    first_profession_index + len(first_profession_tokenized))
                dataset["wrong_profession"].append(second_profession_index - 1)
                dataset["conjunction"].append(
                    second_profession_index + len(second_profession_tokenized))
                dataset["first_pronoun"].append(first_pronoun)
                dataset["circumstances"].append(first_pronoun + 1)
                dataset["dot"].append(tokens_list.index("."))
                dataset["The"].append(tokens_list.index(".") + 1)
                dataset["pronoun"].append(tokens_list.index(".") + 2)
                dataset["second_pronoun"].append(tokens_list.index(
                    ".") + 4 if "Llama" in model_name else tokens_list.index(".") + 3)
                dataset["refers"].append(tokens_list.index(
                    ".") + 5 if "Llama" in model_name else tokens_list.index(".") + 4)
                dataset["to"].append(tokens_list.index(
                    ".") + 6 if "Llama" in model_name else tokens_list.index(".") + 5)
                dataset["the"].append(tokens_list.index(
                    ".") + 7 if "Llama" in model_name else tokens_list.index(".") + 6)
                dataset["length"].append(len(tokens_list))

    dataset["split"] = split
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    path = os.path.join(save_dir, "anti_male_first_prof.csv")
    dataset.to_csv(path)

    eval_model_on_winobias(model_name, save_dir)


@torch.no_grad()
def eval_model_on_winobias(model_name: str, save_dir: str, batch_size: int = 8) -> None:
    """
    Evaluate a model's performance on the WinoBias dataset.

    This function evaluates a model on the WinoBias dataset, which tests for gender bias in 
    profession-based coreference resolution. It processes multiple test sets with different
    configurations (pro/anti-stereotypical, male/female professions) and computes prediction
    probabilities.

    Args:
        model (str): Name/path of the model to evaluate
        batch_size (int, optional): Batch size for processing. Defaults to 8.

    Returns:
        None: Results are processed and stored internally

    The function:
    1. Loads test files from multiple random seeds
    2. Processes prompts in batches
    3. For each example:
        - Gets model predictions and probabilities
        - Records top predicted answer and its probability
        - Records probabilities for correct and wrong profession tokens
    """

    files_list = []
    files_list += [
        os.path.join(save_dir, "anti_female_first_prof.csv"),
        os.path.join(save_dir, "anti_female_second_prof.csv"),
        os.path.join(save_dir, "anti_male_first_prof.csv"),
        os.path.join(save_dir, "anti_male_second_prof.csv"),
        os.path.join(save_dir, "pro_female_first_prof.csv"),
        os.path.join(save_dir, "pro_female_second_prof.csv"),
        os.path.join(save_dir, "pro_male_first_prof.csv"),
        os.path.join(save_dir, "pro_male_second_prof.csv")
    ]

    dtype = "bf16" if "Llama" in model_name else "float32"
    model = HookedTransformer.from_pretrained(
        model,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    model.eval()
    for file in files_list:

        top_answer_list = []
        top_answer_prob_list = []
        correct_prob_list = []
        wrong_prob_list = []
        data = pd.read_csv(file, index_col=[0])
        num_batches = math.ceil(data.shape[0] / batch_size)
        print(file)
        for b in tqdm(range(num_batches)):
            batch = data.iloc[b *
                              batch_size: (b + 1) * batch_size].reset_index()
            logits = model(batch["prompt"].to_list(), return_type="logits")
            for index, row in batch.iterrows():
                outputs_token = torch.argmax(
                    logits[index, row["length"]-1], dim=-1)

                probs = torch.softmax(logits[index, row["length"]-1], dim=-1)
                output_prob = probs[outputs_token].item()
                correct_prob = probs[model.to_single_token(
                    row["correct_token"])].item()
                wrong_prob = probs[model.to_single_token(
                    row["wrong_token"])].item()

                top_answer_list.append(model.to_string(outputs_token))
                top_answer_prob_list.append(round(output_prob, 4))
                correct_prob_list.append(round(correct_prob, 4))
                wrong_prob_list.append(round(wrong_prob, 4))
            del batch, logits
        data = data.assign(top_answer=top_answer_list, top_answer_prob=top_answer_prob_list, correct_prob=correct_prob_list,
                           wrong_prob=wrong_prob_list)

        num_correct = data[data['top_answer']
                           == data['correct_token']].shape[0]
        num_wrong = data[data['top_answer'] == data['wrong_token']].shape[0]
        print("correct:", num_correct / data.shape[0])
        print("wrong:", num_wrong / data.shape[0])

        data.to_csv(file, index=False)


def create_greather_than_dataset(model_name: str, save_dir: str, seed: int = 42) -> None:
    """
    Create a dataset for the greater-than task.


    Args:
        seed (int): Random seed for reproducibility
        model (str): Name of the model being evaluated

    Returns:
        None: Saves generated datasets as CSV files in data/{model_name}/greater_than/{seed}/
        with two files:
        - clean: Original prompts with correct year comparisons
        - corrupted: Modified prompts with swapped years
    """

    dtype = "bf16" if "Llama" in model_name else "float32"
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    df = pd.read_csv("data/gpt2/greater_than/greater_than_data.csv")

    random.seed(seed)
    types = random.choices(["circuit", "ablation", "eval"], weights=[
                           40, 20, 40], k=df.shape[0])
    dataset_clean = defaultdict(list)
    dataset_counter_vanila = defaultdict(list)
    for index, row in df.iterrows():
        tokens = model.to_str_tokens(row["clean"], prepend_bos=True)
        assert len(tokens[8]) == 2
        dataset_clean["prompt"].append(row["clean"])
        dataset_counter_vanila["prompt"].append(row["corrupted"])

        # Define the token positions for each word in the sequence
        token_positions = {
            "The": 1, "NOUN": 2, "lasted": 3, "from": 4,
            "the_1": 5, "year_1": 6, "XX1": 7, "YY": 8,
            "to": 9, "the_2": 10, "year_2": 11, "XX2": 12,
            "length": 13
        }

        # Add token positions to both datasets
        for key, pos in token_positions.items():
            dataset_clean[key].append(pos)
            dataset_counter_vanila[key].append(pos)

        # Add metadata fields
        dataset_clean["label"].append(row["label"])
        dataset_counter_vanila["label"].append("01")

        dataset_clean["split"].append(types[index])
        dataset_counter_vanila["split"].append(types[index])

    dataset_clean = pd.DataFrame.from_dict(dataset_clean).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    dataset_counter_vanila = pd.DataFrame.from_dict(dataset_counter_vanila).sample(
        frac=1, random_state=seed).reset_index(drop=True)

    dataset_clean.to_csv(os.path.join(save_dir, "greater_than_data_clean.csv"))
    dataset_counter_vanila.to_csv(os.path.join(
        save_dir, "greater_than_data_counter_vanila.csv"))

    eval_model_on_gt(model_name, save_dir)


def eval_model_on_gt(model_name: str, save_dir: str, batch_size: int = 8) -> None:
    """
    Evaluate a model's performance on the greater-than task.

    This function evaluates a model on the greater-than task by processing datasets containing prompts.
    It computes probabilities for the model's predictions and saves evaluation metrics including top answers
    and their probabilities.

    """

    file = os.path.join(os.path.join(save_dir, "greater_than_data_clean.csv"))

    dtype = "bf16" if "Llama" else "float32"
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    model.eval()

    correctness_list = []
    num_correct, num_wrong = 0, 0
    data = pd.read_csv(file, index_col=[0])
    num_batches = math.ceil(data.shape[0] / batch_size)
    print(file)
    for b in tqdm(range(num_batches)):
        batch = data.iloc[b * batch_size: (b + 1) * batch_size].reset_index()
        logits = model(batch["prompt"].to_list(), return_type="logits")
        for index, row in batch.iterrows():
            outputs_token = torch.argmax(logits[index, -1], dim=-1)
            if int(model.tokenizer.decode(outputs_token)) >= row["label"]:
                num_correct += 1
                correctness_list.append(1)
            else:
                num_wrong += 1
                correctness_list.append(0)
    data = data.assign(is_model_correct=correctness_list)

    print("correct:", num_correct / data.shape[0])
    print("wrong:", num_wrong / data.shape[0])

    data.to_csv(file, index=False)


def create_IOI_dataset_ABBA(model_name: str, save_dir: str, seed: int = 42) -> None:
    """
    Generate IOI (Indirect Object Identification) dataset in ABBA format.

    This function creates a dataset of sentences following the Indirect Object Identification (IOI) 
    pattern in ABBA format, where names are arranged in an ABBA pattern (e.g., "Name1 Name2 Name2 Name1").
    The task is to identify the correct referent in sentences with this structure.

    Args:
        model_name (str): Name of the model being evaluated

    Returns:
        None: Saves generated datasets as CSV files in data/{model_name}/ioi/{seed}/ directories

    """
    NAMES_gpt2 = [
        "Michael",
        "Christopher",
        "Jessica",
        "Matthew",
        "Ashley",
        "Jennifer",
        "Joshua",
        "Amanda",
        "Daniel",
        "David",
        "James",
        "Robert",
        "John",
        "Joseph",
        "Andrew",
        "Ryan",
        "Brandon",
        "Jason",
        "Justin",
        "Sarah",
        "William",
        "Jonathan",
        "Stephanie",
        "Brian",
        "Nicole",
        "Nicholas",
        "Anthony",
        "Heather",
        "Eric",
        "Elizabeth",
        "Adam",
        "Megan",
        "Melissa",
        "Kevin",
        "Steven",
        "Thomas",
        "Timothy",
        "Christina",
        "Kyle",
        "Rachel",
        "Laura",
        "Lauren",
        "Amber",
        "Brittany",
        "Danielle",
        "Richard",
        "Kimberly",
        "Jeffrey",
        "Amy",
        "Crystal",
        "Michelle",
        "Tiffany",
        "Jeremy",
        "Benjamin",
        "Mark",
        "Emily",
        "Aaron",
        "Charles",
        "Rebecca",
        "Jacob",
        "Stephen",
        "Patrick",
        "Sean",
        "Erin",
        "Jamie",
        "Kelly",
        "Samantha",
        "Nathan",
        "Sara",
        "Dustin",
        "Paul",
        "Angela",
        "Tyler",
        "Scott",
        "Katherine",
        "Andrea",
        "Gregory",
        "Erica",
        "Mary",
        "Travis",
        "Lisa",
        "Kenneth",
        "Bryan",
        "Lindsey",
        "Kristen",
        "Jose",
        "Alexander",
        "Jesse",
        "Katie",
        "Lindsay",
        "Shannon",
        "Vanessa",
        "Courtney",
        "Christine",
        "Alicia",
        "Cody",
        "Allison",
        "Bradley",
        "Samuel",
    ]

    NAMES_llam3 = [
        "Michael",
        "Christopher",
        "Jessica",
        "Matthew",
        "Jennifer",
        "Daniel",
        "David",
        "James",
        "Robert",
        "John",
        "Joseph",
        "Andrew",
        "Ryan",
        "Brandon",
        "Jason",
        "Justin",
        "Sarah",
        "William",
        "Jonathan",
        "Brian",
        "Anthony",
        "Eric",
        "Elizabeth",
        "Adam",
        "Kevin",
        "Steven",
        "Thomas",
        "Kyle",
        "Rachel",
        "Laura",
        "Richard",
        "Amy",
        "Crystal",
        "Michelle",
        "Jeremy",
        "Mark",
        "Emily",
        "Aaron",
        "Charles",
        "Jacob",
        "Stephen",
        "Patrick",
        "Sean",
        "Jamie",
        "Kelly",
        "Paul",
        "Tyler",
        "Scott",
        "Mary",
        "Lisa",
        "Jose",
        "Alexander",
    ]

    PLACES = [
        "store",
        "garden",
        "restaurant",
        "school",
        "hospital",
        "office",
        "house",
        "station",
    ]
    OBJECTS = [
        "ring",
        "kiss",
        "bone",
        "basketball",
        "computer",
        "necklace",
        "drink",
        "snack",
    ]

    BABA_TEMPLATES = [
        "Then, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to",
        "Then, [A] and [B] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to",
        "Then, [A] and [B] were working at the [PLACE]. [B] decided to give a [OBJECT] to",
        "Then, [A] and [B] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to",
        "Then, [A] and [B] had a long argument, and afterwards [B] said to",
        "After [A] and [B] went to the [PLACE], [B] gave a [OBJECT] to",
        "When [A] and [B] got a [OBJECT] at the [PLACE], [B] decided to give it to",
        "When [A] and [B] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to",
        "While [A] and [B] were working at the [PLACE], [B] gave a [OBJECT] to",
        "While [A] and [B] were commuting to the [PLACE], [B] gave a [OBJECT] to",
        "After the lunch, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to",
        "Afterwards, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to",
        "Then, [A] and [B] had a long argument. Afterwards [B] said to",
        "The [PLACE] [A] and [B] went to had a [OBJECT]. [B] gave it to",
        "Friends [A] and [B] found a [OBJECT] at the [PLACE]. [B] gave it to",
    ]

    ABC_TEMPLATES = [
        "Then, [B] and [A] went to the [PLACE]. [C] gave a [OBJECT] to",
        "Then, [B] and [A] had a lot of fun at the [PLACE]. [C] gave a [OBJECT] to",
        "Then, [B] and [A] were working at the [PLACE]. [C] decided to give a [OBJECT] to",
        "Then, [B] and [A] were thinking about going to the [PLACE]. [C] wanted to give a [OBJECT] to",
        "Then, [B] and [A] had a long argument, and afterwards [C] said to",
        "After [B] and [A] went to the [PLACE], [C] gave a [OBJECT] to",
        "When [B] and [A] got a [OBJECT] at the [PLACE], [C] decided to give it to",
        "When [B] and [A] got a [OBJECT] at the [PLACE], [C] decided to give the [OBJECT] to",
        "While [B] and [A] were working at the [PLACE], [C] gave a [OBJECT] to",
        "While [B] and [A] were commuting to the [PLACE], [C] gave a [OBJECT] to",
        "After the lunch, [B] and [A] went to the [PLACE]. [C] gave a [OBJECT] to",
        "Afterwards, [B] and [A] went to the [PLACE]. [C] gave a [OBJECT] to",
        "Then, [B] and [A] had a long argument. Afterwards [C] said to",
        "The [PLACE] [B] and [A] went to had a [OBJECT]. [C] gave it to",
        "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [C] gave it to",
    ]

    BABA_FULL_TEMPLATES = []
    ABC_FULL_TEMPLATES = []

    for template in BABA_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                BABA_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))
    for template in ABC_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                ABC_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))

    dtype = "bf16" if "Llama" else "float32"
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    NAMES = NAMES_gpt2 if model_name == "gpt2" else NAMES_llam3
    names_comb = list(itertools.combinations(NAMES, 5))
    dataset_size = 30000
    print("comb:", len(names_comb))
    random.seed(seed)
    types = random.choices(["circuit", "eval", "ablation"], weights=[
                           10, 10, 80], k=len(names_comb))
    dataset_clean = defaultdict(list)
    dataset_counter_abc = defaultdict(list)
    names_comb_seed = random.sample(names_comb, dataset_size)
    for i in tqdm(range(len(names_comb_seed))):
        s_token, io_token, a_token, b_token, c_token = names_comb_seed[i]
        template_index = random.randint(0, len(BABA_FULL_TEMPLATES) - 1)
        baba_prompt = BABA_FULL_TEMPLATES[template_index].replace(
            "[A]", io_token).replace("[B]", s_token)
        tokens_list = model.to_str_tokens(baba_prompt, prepend_bos=True)
        io_index = tokens_list.index(" " + io_token)
        s1_index = tokens_list.index(" " + s_token)
        s2_index = tokens_list[s1_index +
                               1:].index(" " + s_token) + s1_index + 1
        dataset_clean["prompt"].append(baba_prompt)
        dataset_clean["prompt_id"].append(template_index)
        dataset_clean["prefix"].append(1)
        dataset_clean["IO"].append(io_index)
        dataset_clean["and"].append(io_index + 1)
        dataset_clean["S1"].append(s1_index)
        dataset_clean["S1+1"].append(s1_index+1)
        dataset_clean["action1"].append(s1_index + 2)
        dataset_clean["S2"].append(s2_index)
        dataset_clean["action2"].append(s2_index + 1)
        dataset_clean["to"].append(len(tokens_list) - 1)
        dataset_clean["length"].append(len(tokens_list))
        dataset_clean["wrong_token"].append(" " + s_token)
        dataset_clean["correct_token"].append(" " + io_token)
        dataset_clean["S1_token"].append(" " + s_token)
        dataset_clean["S2_token"].append(" " + s_token)
        dataset_clean["IO_token"].append(" " + io_token)
        dataset_clean["label"].append(" " + io_token)
        dataset_clean["split"].append(types[i])

        abc_prompt = ABC_FULL_TEMPLATES[template_index].replace(
            "[A]", a_token).replace("[B]", b_token).replace("[C]", c_token)

        dataset_counter_abc["prompt"].append(abc_prompt)
        dataset_counter_abc["prompt_id"].append(template_index)
        dataset_counter_abc["prefix"].append(1)
        dataset_counter_abc["IO"].append(io_index)
        dataset_counter_abc["and"].append(io_index + 1)
        dataset_counter_abc["S1"].append(s1_index)
        dataset_counter_abc["S1+1"].append(s1_index+1)
        dataset_counter_abc["action1"].append(s1_index + 2)
        dataset_counter_abc["S2"].append(s2_index)
        dataset_counter_abc["action2"].append(s2_index + 1)
        dataset_counter_abc["to"].append(len(tokens_list) - 1)
        dataset_counter_abc["length"].append(len(tokens_list))
        dataset_counter_abc["wrong_token"].append(" " + s_token)
        dataset_counter_abc["correct_token"].append(" " + io_token)
        dataset_counter_abc["S1_token"].append(" " + s_token)
        dataset_counter_abc["S2_token"].append(" " + s_token)
        dataset_counter_abc["IO_token"].append(" " + io_token)
        dataset_counter_abc["label"].append(" " + io_token)
        dataset_counter_abc["split"].append(types[i])

    dataset_clean = pd.DataFrame.from_dict(dataset_clean)
    print("data size:", dataset_clean.shape[0])
    dataset_clean = dataset_clean.drop_duplicates()

    dataset_counter_abc = pd.DataFrame.from_dict(dataset_counter_abc)
    dataset_counter_abc = dataset_counter_abc[dataset_counter_abc.index.isin(
        dataset_clean.index)]

    dataset_clean = dataset_clean.sample(
        frac=1, random_state=seed).reset_index(drop=True)
    dataset_counter_abc = dataset_counter_abc.sample(
        frac=1, random_state=seed).reset_index(drop=True)

    dataset_clean.to_csv(os.path.join(save_dir, f'IOI_ABBA_data_clean.csv'))
    dataset_counter_abc.to_csv(os.path.join(
        save_dir, f'IOI_ABBA_data_counter_abc.csv'))

    eval_model_on_ioi(model_name=model_name,
                      type_dataset="ABBA", save_dir=save_dir)


def create_IOI_dataset_BABA(model_name: str, save_dir: str, seed: int = 42) -> None:
    """
    Generate IOI (Indirect Object Identification) dataset in BABA format.

    This function creates a dataset of sentences following the Indirect Object Identification (IOI)
    pattern in BABA format, where names are arranged in a BABA pattern (e.g., "Name2 Name1 Name2 Name1"). 
    The task is to identify the correct referent in sentences with this structure.

    Args:
        model_name (str): Name of the model being evaluated

    Returns:
        None: Saves generated datasets as CSV files in data/{model_name}/ioi/{seed}/ directories
    """
    NAMES_gpt2 = [
        "Michael",
        "Christopher",
        "Jessica",
        "Matthew",
        "Ashley",
        "Jennifer",
        "Joshua",
        "Amanda",
        "Daniel",
        "David",
        "James",
        "Robert",
        "John",
        "Joseph",
        "Andrew",
        "Ryan",
        "Brandon",
        "Jason",
        "Justin",
        "Sarah",
        "William",
        "Jonathan",
        "Stephanie",
        "Brian",
        "Nicole",
        "Nicholas",
        "Anthony",
        "Heather",
        "Eric",
        "Elizabeth",
        "Adam",
        "Megan",
        "Melissa",
        "Kevin",
        "Steven",
        "Thomas",
        "Timothy",
        "Christina",
        "Kyle",
        "Rachel",
        "Laura",
        "Lauren",
        "Amber",
        "Brittany",
        "Danielle",
        "Richard",
        "Kimberly",
        "Jeffrey",
        "Amy",
        "Crystal",
        "Michelle",
        "Tiffany",
        "Jeremy",
        "Benjamin",
        "Mark",
        "Emily",
        "Aaron",
        "Charles",
        "Rebecca",
        "Jacob",
        "Stephen",
        "Patrick",
        "Sean",
        "Erin",
        "Jamie",
        "Kelly",
        "Samantha",
        "Nathan",
        "Sara",
        "Dustin",
        "Paul",
        "Angela",
        "Tyler",
        "Scott",
        "Katherine",
        "Andrea",
        "Gregory",
        "Erica",
        "Mary",
        "Travis",
        "Lisa",
        "Kenneth",
        "Bryan",
        "Lindsey",
        "Kristen",
        "Jose",
        "Alexander",
        "Jesse",
        "Katie",
        "Lindsay",
        "Shannon",
        "Vanessa",
        "Courtney",
        "Christine",
        "Alicia",
        "Cody",
        "Allison",
        "Bradley",
        "Samuel",
    ]

    NAMES_llam3 = [
        "Michael",
        "Christopher",
        "Jessica",
        "Matthew",
        "Jennifer",
        "Daniel",
        "David",
        "James",
        "Robert",
        "John",
        "Joseph",
        "Andrew",
        "Ryan",
        "Brandon",
        "Jason",
        "Justin",
        "Sarah",
        "William",
        "Jonathan",
        "Brian",
        "Anthony",
        "Eric",
        "Elizabeth",
        "Adam",
        "Kevin",
        "Steven",
        "Thomas",
        "Kyle",
        "Rachel",
        "Laura",
        "Richard",
        "Amy",
        "Crystal",
        "Michelle",
        "Jeremy",
        "Mark",
        "Emily",
        "Aaron",
        "Charles",
        "Jacob",
        "Stephen",
        "Patrick",
        "Sean",
        "Jamie",
        "Kelly",
        "Paul",
        "Tyler",
        "Scott",
        "Mary",
        "Lisa",
        "Jose",
        "Alexander",
    ]

    PLACES = [
        "store",
        "garden",
        "restaurant",
        "school",
        "hospital",
        "office",
        "house",
        "station",
    ]
    OBJECTS = [
        "ring",
        "kiss",
        "bone",
        "basketball",
        "computer",
        "necklace",
        "drink",
        "snack",
    ]

    BABA_TEMPLATES = [
        "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
        "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to",
        "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to",
        "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to",
        "Then, [B] and [A] had a long argument, and afterwards [B] said to",
        "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to",
        "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to",
        "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to",
        "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to",
        "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to",
        "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
        "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
        "Then, [B] and [A] had a long argument. Afterwards [B] said to",
        "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to",
        "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to",
    ]

    ABC_TEMPLATES = [
        "Then, [B] and [A] went to the [PLACE]. [C] gave a [OBJECT] to",
        "Then, [B] and [A] had a lot of fun at the [PLACE]. [C] gave a [OBJECT] to",
        "Then, [B] and [A] were working at the [PLACE]. [C] decided to give a [OBJECT] to",
        "Then, [B] and [A] were thinking about going to the [PLACE]. [C] wanted to give a [OBJECT] to",
        "Then, [B] and [A] had a long argument, and afterwards [C] said to",
        "After [B] and [A] went to the [PLACE], [C] gave a [OBJECT] to",
        "When [B] and [A] got a [OBJECT] at the [PLACE], [C] decided to give it to",
        "When [B] and [A] got a [OBJECT] at the [PLACE], [C] decided to give the [OBJECT] to",
        "While [B] and [A] were working at the [PLACE], [C] gave a [OBJECT] to",
        "While [B] and [A] were commuting to the [PLACE], [C] gave a [OBJECT] to",
        "After the lunch, [B] and [A] went to the [PLACE]. [C] gave a [OBJECT] to",
        "Afterwards, [B] and [A] went to the [PLACE]. [C] gave a [OBJECT] to",
        "Then, [B] and [A] had a long argument. Afterwards [C] said to",
        "The [PLACE] [B] and [A] went to had a [OBJECT]. [C] gave it to",
        "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [C] gave it to",
    ]

    BABA_FULL_TEMPLATES = []
    ABC_FULL_TEMPLATES = []

    for template in BABA_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                BABA_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))
    for template in ABC_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                ABC_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))

    dtype = "bf16" if "Llama" else "float32"
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    NAMES = NAMES_gpt2 if model_name == "gpt2" else NAMES_llam3
    names_comb = list(itertools.combinations(NAMES, 5))
    dataset_size = 30000
    print("comb:", len(names_comb))
    random.seed(seed)
    types = random.choices(["circuit", "eval", "ablation"], weights=[
                           10, 10, 80], k=len(names_comb))
    dataset_clean = defaultdict(list)
    dataset_counter_abc = defaultdict(list)
    names_comb_seed = random.sample(names_comb, dataset_size)
    for i in tqdm(range(len(names_comb_seed))):
        s_token, io_token, a_token, b_token, c_token = names_comb_seed[i]
        template_index = random.randint(0, len(BABA_FULL_TEMPLATES) - 1)
        baba_prompt = BABA_FULL_TEMPLATES[template_index].replace(
            "[A]", io_token).replace("[B]", s_token)
        tokens_list = model.to_str_tokens(baba_prompt, prepend_bos=True)
        io_index = tokens_list.index(" " + io_token)
        s1_index = tokens_list.index(" " + s_token)
        s2_index = tokens_list[s1_index +
                               1:].index(" " + s_token) + s1_index + 1
        dataset_clean["prompt"].append(baba_prompt)
        dataset_clean["prompt_id"].append(template_index)
        dataset_clean["prefix"].append(1)
        dataset_clean["S1"].append(s1_index)
        dataset_clean["S1+1"].append(s1_index + 1)
        dataset_clean["IO"].append(io_index)
        dataset_clean["action1"].append(io_index + 1)
        dataset_clean["S2"].append(s2_index)
        dataset_clean["action2"].append(s2_index + 1)
        dataset_clean["to"].append(len(tokens_list) - 1)
        dataset_clean["length"].append(len(tokens_list))
        dataset_clean["wrong_token"].append(" " + s_token)
        dataset_clean["correct_token"].append(" " + io_token)
        dataset_clean["S1_token"].append(" " + s_token)
        dataset_clean["S2_token"].append(" " + s_token)
        dataset_clean["IO_token"].append(" " + io_token)
        dataset_clean["label"].append(" " + io_token)
        dataset_clean["split"].append(types[i])

        abc_prompt = ABC_FULL_TEMPLATES[template_index].replace(
            "[A]", a_token).replace("[B]", b_token).replace("[C]", c_token)

        dataset_counter_abc["prompt"].append(abc_prompt)
        dataset_counter_abc["prompt_id"].append(template_index)
        dataset_counter_abc["prefix"].append(1)
        dataset_counter_abc["S1"].append(s1_index)
        dataset_counter_abc["S1+1"].append(s1_index + 1)
        dataset_counter_abc["IO"].append(io_index)
        dataset_counter_abc["action1"].append(io_index + 1)
        dataset_counter_abc["S2"].append(s2_index)
        dataset_counter_abc["action2"].append(s2_index + 1)
        dataset_counter_abc["to"].append(len(tokens_list) - 1)
        dataset_counter_abc["length"].append(len(tokens_list))
        dataset_counter_abc["wrong_token"].append(" " + s_token)
        dataset_counter_abc["correct_token"].append(" " + io_token)
        dataset_counter_abc["S1_token"].append(" " + s_token)
        dataset_counter_abc["S2_token"].append(" " + s_token)
        dataset_counter_abc["IO_token"].append(" " + io_token)
        dataset_counter_abc["label"].append(" " + io_token)
        dataset_counter_abc["split"].append(types[i])

    dataset_clean = pd.DataFrame.from_dict(dataset_clean)
    print("data size:", dataset_clean.shape[0])
    dataset_clean = dataset_clean.drop_duplicates()

    dataset_counter_abc = pd.DataFrame.from_dict(dataset_counter_abc)
    dataset_counter_abc = dataset_counter_abc[dataset_counter_abc.index.isin(
        dataset_clean.index)]

    dataset_clean = dataset_clean.sample(
        frac=1, random_state=seed).reset_index(drop=True)
    dataset_counter_abc = dataset_counter_abc.sample(
        frac=1, random_state=seed).reset_index(drop=True)

    dataset_clean.to_csv(os.path.join(save_dir, f'IOI_BABA_data_clean.csv'))
    dataset_counter_abc.to_csv(os.path.join(
        save_dir, f'IOI_BABA_data_counter_abc.csv'))

    eval_model_on_ioi(model_name, type_dataset="BABA",
                      save_dir=save_dir, batch_size=8)


def create_IOI_jp_dataset_ABBA(model_name: str, save_dir: str, seed: int = 42) -> None:
    """
    Generate IOI (Indirect Object Identification) japanese dataset in ABBA format.

    This function creates a dataset of sentences following the Indirect Object Identification (IOI)
    pattern in ABBA format, where names are arranged in an ABBA pattern (e.g., "Name2 Name1 Name1 Name2").
    The task is to identify the correct referent in sentences with this structure.

    Args:
        model_name (str): Name of the model being evaluated

    Returns:
        None: Saves generated datasets as CSV files in data/{model_name}/ioi/{seed}/ directories
    """

    JAPANESE_NAMES = [
        "翔太", "美咲", "大輔", "陽菜", "健太",
        "結衣", "拓也", "愛", "直人", "未来",
        "亮", "さくら", "哲也", "美優", "達也",
        "七海", "一輝", "葵", "翼", "美月",
        "和也", "楓", "涼太", "優花", "直樹",
        "彩", "剛", "優菜", "隼人", "里奈",
        "陸", "美羽", "智也", "花", "蓮",
        "杏奈", "聡", "千尋", "裕太", "美穂",
        "誠", "遥", "大樹", "真央", "修", "光", "加奈", "雄大", "香織"
    ]

    PLACES = [
        "店",
        "庭",
        "レストラン",
        "学校",
        "病院",
        "オフィス",
        "家",
        "駅",
        "山",
    ]
    OBJECTS = [
        "指輪",
        "キス",
        "骨",
        "バスケットボール",
        "コンピューター",
        "ネックレス",
        "飲み物",
        "スナック",
    ]

    ABBA_TEMPLATES = [
        # 1. Passing an item
        "[PLACE]で[A]と[B]が話していた。[B]は[OBJECT]を取り出すと、",

        # 2. Returning a lost item
        "[A]と[B]は[PLACE]にいた。[B]は落ちていた[OBJECT]を拾い、",

        # 3. Showing a discovery
        "[PLACE]で[A]と[B]が歩いていたとき、[B]は珍しい[OBJECT]を見つけ、すぐに",

        # 4. Buying a gift
        "[PLACE]で[A]と[B]が買い物をしていた。[B]は素敵な[OBJECT]を購入し、",

        # 5. Passing a ball (Sports context)
        "[PLACE]で[A]と[B]が練習していた。[B]は[OBJECT]をキャッチすると、力強く",

        # 6. Serving food/drink
        "[A]と[B]が[PLACE]で席に着いた。[B]は熱い[OBJECT]を持ってきて、",

        # 7. Explaining a document
        "[PLACE]で[A]と[B]が仕事をしていた。[B]は新しい[OBJECT]を作成し、その内容を",

        # 8. Sharing a secret
        "[PLACE]の隅で[A]と[B]が二人きりだった。[B]は衝撃的な[OBJECT]を知り、こっそりと",

        # 9. Recommending a book/item
        "[PLACE]で[A]と[B]が本棚を見ていた。[B]は面白そうな[OBJECT]を選び、",

        # 10. Handing over a tool
        "[A]と[B]は[PLACE]で修理をしていた。[B]は近くにあった[OBJECT]を掴み、",

        # 11. Lending a spare item
        "[PLACE]で[A]と[B]が準備をしていた。[B]は余分な[OBJECT]を持っていたので、",

        # 12. Showing a photo/screen
        "[A]と[B]は[PLACE]で休憩していた。[B]は携帯の画面にある[OBJECT]を表示し、",

        # 13. Giving a ride/key (Leaving context)
        "[PLACE]で[A]と[B]が帰り支度をしていた。[B]は車の[OBJECT]を見つけ、",

        # 14. Returning a borrowed book
        "[A]と[B]は[PLACE]の図書館にいた。[B]は借りていた[OBJECT]を読み終え、",

        # 15. Pouring a drink (Party context)
        "[PLACE]で[A]と[B]が乾杯しようとしていた。[B]は冷えた[OBJECT]を開け、"
    ]

    ABC_TEMPLATES = [
        # Replace second [B] with [C]
        template[:template.find('[B]', template.find('[B]') + 1)] + '[C]' +
        template[template.find('[B]', template.find('[B]') + 1) + len('[B]'):]
        for template in ABBA_TEMPLATES
    ]

    ABBA_FULL_TEMPLATES = []
    ABC_FULL_TEMPLATES = []

    for template in ABBA_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                ABBA_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))
    for template in ABC_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                ABC_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))

    dtype = "bf16" if "Llama" else "float32"
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        # default_prepend_bos=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    names_comb = [c for c in itertools.combinations(JAPANESE_NAMES, 5)
                  #   Require different start token for s1 and IO, and same length for clean and counterfactual
                  if c[0][0] != c[1][0] and
                  2*len(c[0]) + len(c[1]) == len(c[2]) + len(c[3]) + len(c[4])]
    dataset_size = 30000
    print("comb:", len(names_comb))
    random.seed(seed)
    types = random.choices(["circuit", "eval", "ablation"], weights=[
                           10, 10, 80], k=len(names_comb))
    dataset_clean = defaultdict(list)
    dataset_counter_abc = defaultdict(list)
    names_comb_seed = random.sample(names_comb, dataset_size)
    for i in tqdm(range(len(names_comb_seed))):
        s_token, io_token, a_token, b_token, c_token = names_comb_seed[i]
        template_index = random.randint(0, len(ABBA_FULL_TEMPLATES) - 1)
        baba_prompt = ABBA_FULL_TEMPLATES[template_index].replace(
            "[A]", io_token).replace("[B]", s_token)
        tokens_list = model.to_str_tokens(baba_prompt, prepend_bos=True)
        abc_prompt = ABC_FULL_TEMPLATES[template_index].replace(
            "[A]", a_token).replace("[B]", b_token).replace("[C]", c_token)
        abc_tokens_list = model.to_str_tokens(abc_prompt, prepend_bos=True)
        if len(tokens_list) != len(abc_tokens_list):
            continue

        io_tokens = model.to_str_tokens(io_token)
        s_tokens = model.to_str_tokens(s_token)
        io_index = tokens_list.index(io_tokens[0])
        s1_index = tokens_list.index(s_tokens[0])
        s2_index = tokens_list[s1_index +
                               1:].index(s_tokens[0]) + s1_index + 1
        dataset_clean["prompt"].append(baba_prompt)
        dataset_clean["prompt_id"].append(template_index)
        dataset_clean["prefix"].append(1)
        dataset_clean["IO"].append(io_index)
        dataset_clean["connector"].append(io_index + len(io_tokens))
        dataset_clean["S1"].append(s1_index)
        dataset_clean["action1"].append(s1_index + len(s_tokens))
        dataset_clean["S2"].append(s2_index)
        dataset_clean["action2"].append(s2_index + len(s_tokens))
        # Not really "to" in Japanese, but for consistency
        dataset_clean["to"].append(len(tokens_list) - 1)
        dataset_clean["length"].append(len(tokens_list))
        dataset_clean["wrong_token"].append(s_tokens[0])
        dataset_clean["correct_token"].append(io_tokens[0])
        dataset_clean["S1_token"].append(s_tokens[0])
        dataset_clean["S2_token"].append(s_tokens[0])
        dataset_clean["IO_token"].append(io_tokens[0])
        dataset_clean["label"].append(io_tokens[0])
        dataset_clean["split"].append(types[i])

        a_tokens = model.to_str_tokens(a_token)
        b_tokens = model.to_str_tokens(b_token)
        c_tokens = model.to_str_tokens(c_token)
        a_index = abc_tokens_list.index(a_tokens[0])
        b_index = abc_tokens_list.index(b_tokens[0])
        c_index = abc_tokens_list.index(c_tokens[0])

        dataset_counter_abc["prompt"].append(abc_prompt)
        dataset_counter_abc["prompt_id"].append(template_index)
        dataset_counter_abc["prefix"].append(1)
        dataset_counter_abc["IO"].append(a_index)
        dataset_counter_abc["connector"].append(a_index + len(a_tokens))
        dataset_counter_abc["S1"].append(b_index)
        dataset_counter_abc["action1"].append(b_index + len(b_tokens))
        dataset_counter_abc["S2"].append(c_index)
        dataset_counter_abc["action2"].append(c_index + len(c_tokens))
        dataset_counter_abc["to"].append(len(tokens_list) - 1)
        dataset_counter_abc["length"].append(len(tokens_list))
        dataset_counter_abc["wrong_token"].append(s_tokens[0])
        dataset_counter_abc["correct_token"].append(io_tokens[0])
        dataset_counter_abc["S1_token"].append(s_tokens[0])
        dataset_counter_abc["S2_token"].append(s_tokens[0])
        dataset_counter_abc["IO_token"].append(io_tokens[0])
        dataset_counter_abc["label"].append(io_tokens[0])
        dataset_counter_abc["split"].append(types[i])

    dataset_clean = pd.DataFrame.from_dict(dataset_clean)
    print("data size:", dataset_clean.shape[0])
    dataset_clean = dataset_clean.drop_duplicates()

    dataset_counter_abc = pd.DataFrame.from_dict(dataset_counter_abc)
    dataset_counter_abc = dataset_counter_abc[dataset_counter_abc.index.isin(
        dataset_clean.index)]

    dataset_clean = dataset_clean.sample(
        frac=1, random_state=seed).reset_index(drop=True)
    dataset_counter_abc = dataset_counter_abc.sample(
        frac=1, random_state=seed).reset_index(drop=True)

    dataset_clean.to_csv(os.path.join(save_dir, f'IOI_ABBA_jp_data_clean.csv'))
    dataset_counter_abc.to_csv(os.path.join(
        save_dir, f'IOI_ABBA_jp_data_counter_abc.csv'))

    eval_model_on_ioi(model_name, type_dataset="ABBA_jp",
                      save_dir=save_dir, batch_size=8)

def find_sublist_index(main_list, sub_list):
    """Returns first index of sub_list of main_list. Otherwise, returns -1."""
    n = len(main_list)
    m = len(sub_list)
    for i in range(n - m + 1):
        if main_list[i:i+m] == sub_list:
            return i
    return -1

def create_IOI_tr_dataset_ABBA(model_name: str, save_dir: str, seed: int = 42) -> None:
    """
    Generate IOI (Indirect Object Identification) japanese dataset in ABBA format.

    This function creates a dataset of sentences following the Indirect Object Identification (IOI)
    pattern in ABBA format, where names are arranged in an ABBA pattern (e.g., "Name2 Name1 Name1 Name2").
    The task is to identify the correct referent in sentences with this structure.

    Args:
        model_name (str): Name of the model being evaluated

    Returns:
        None: Saves generated datasets as CSV files in data/{model_name}/ioi/{seed}/ directories
    """

    TURKISH_NAMES = [
        "Ahmet", "Ali", "Alp", "Arda", "Berk", 
        "Burak", "Can", "Cem", "Deniz", "Ege", 
        "Emre", "Hakan", "Hasan", "Hüseyin", "İbrahim", 
        "Kaan", "Kemal", "Kerem", "Mehmet", "Mert", 
        "Murat", "Mustafa", "Onur", "Ozan", "Tarık",
        "Aslı", "Aylin", "Ayşe", "Büşra", "Cansu", 
        "Ceren", "Derya", "Ece", "Elif", "Esra", 
        "Fatma", "Gizem", "Gözde", "İrem", "Melis", 
        "Merve", "Özge", "Pelin", "Pınar", "Seda", 
        "Selin", "Sinem", "Tuğba", "Yasemin", "Zeynep"
    ]

    PLACES = [
        "Okul", "Park", "Kütüphane", "Kafe", "Market", 
        "Hastane", "Sinema", "Banka", "Ofis", "Müze"
    ]
    OBJECTS = [
        "Kitap", "Anahtar", "Kalem", "Telefon", "Dosya", 
        "Mektup", "Hediye", "Paket", "Çiçek", "Cüzdan"
    ]

    ABBA_TEMPLATES = [
        "Sonra [A] ve [B] [PLACE]'e gittiler. [B] elindeki [OBJECT]'i doğrudan",
        "Sabah [A] ile [B] [PLACE]'te buluştular. [B] çantasından çıkardığı [OBJECT]'i",
        "Dün [A] ve [B] [PLACE]'e yürüdüler. [B] yolda bulduğu [OBJECT]'i hemen",
        "Bugün [A] ile [B] [PLACE] civarındaydı. [B] yeni aldığı [OBJECT]'i göstermek için",
        "Öğleden sonra [A] ve [B] [PLACE]'e vardılar. [B] taşıdığı ağır [OBJECT]'i",
        "Akşam [A] ile [B] [PLACE]'te oturuyorlardı. [B] masanın üzerindeki [OBJECT]'i yavaşça",
        "Ertesi gün [A] ve [B] [PLACE] kapısındaydılar. [B] cebinde sakladığı [OBJECT]'i",
        "Hafta sonu [A] ile [B] [PLACE]'e geziye gittiler. [B] hatıra olarak aldığı [OBJECT]'i",
        "Nihayet [A] ve [B] [PLACE] içinde karşılaştılar. [B] dikkatlice tuttuğu [OBJECT]'i",
        "Gece [A] ile [B] [PLACE] etrafında dolaşıyordu. [B] parlayan [OBJECT]'i işaret ederek",
    ]

    ABC_TEMPLATES = [
        # Replace second [B] with [C]
        template[:template.find('[B]', template.find('[B]') + 1)] + '[C]' +
        template[template.find('[B]', template.find('[B]') + 1) + len('[B]'):]
        for template in ABBA_TEMPLATES
    ]

    ABBA_FULL_TEMPLATES = []
    ABC_FULL_TEMPLATES = []

    for template in ABBA_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                ABBA_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))
    for template in ABC_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                ABC_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))

    dtype = "bf16" if "Llama" else "float32"
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        # default_prepend_bos=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    names_comb = [c for c in itertools.combinations(TURKISH_NAMES, 5)
                  # Require different start token for s1 and IO, and same length for clean and counterfactual
                  if c[0][0] != c[1][0] and
                  2*len(c[0]) + len(c[1]) == len(c[2]) + len(c[3]) + len(c[4])]
    dataset_size = 30000
    print("comb:", len(names_comb))
    random.seed(seed)
    types = random.choices(["circuit", "eval", "ablation"], weights=[
                           10, 10, 80], k=len(names_comb))
    dataset_clean = defaultdict(list)
    dataset_counter_abc = defaultdict(list)
    names_comb_seed = random.sample(names_comb, dataset_size)
    for i in tqdm(range(len(names_comb_seed))):
        s_token, io_token, a_token, b_token, c_token = names_comb_seed[i]
        template_index = random.randint(0, len(ABBA_FULL_TEMPLATES) - 1)
        baba_prompt = ABBA_FULL_TEMPLATES[template_index].replace(
            "[A]", io_token).replace("[B]", s_token)
        tokens_list = model.to_str_tokens(baba_prompt, prepend_bos=True)
        abc_prompt = ABC_FULL_TEMPLATES[template_index].replace(
            "[A]", a_token).replace("[B]", b_token).replace("[C]", c_token)
        abc_tokens_list = model.to_str_tokens(abc_prompt, prepend_bos=True)
        if len(tokens_list) != len(abc_tokens_list):
            continue

        io_tokens = model.to_str_tokens(" " + io_token)
        s_tokens = model.to_str_tokens(" " + s_token)
        io_index = find_sublist_index(tokens_list, io_tokens)
        if io_index == -1:
            print(f"IO token '{io_token}' not found in tokens list for prompt: {baba_prompt}")
            print(f"Tokens list: {tokens_list}, IO token list: {io_tokens}")
            continue
        s1_index = find_sublist_index(tokens_list, s_tokens)
        if s1_index == -1:
            print(f"S1 token '{s_token}' not found in tokens list for prompt: {baba_prompt}")
            print(f"Tokens list: {tokens_list}, S1 token list: {s_tokens}")
            continue
        s2_index = find_sublist_index(tokens_list[s1_index + 1:], s_tokens) + s1_index + 1
        dataset_clean["prompt"].append(baba_prompt)
        dataset_clean["prompt_id"].append(template_index)
        dataset_clean["prefix"].append(1)
        dataset_clean["IO"].append(io_index)
        dataset_clean["connector"].append(io_index + len(io_tokens))
        dataset_clean["S1"].append(s1_index)
        dataset_clean["action1"].append(s1_index + len(s_tokens))
        dataset_clean["S2"].append(s2_index)
        dataset_clean["action2"].append(s2_index + len(s_tokens))
        # Not really "to" in Turkish, but for consistency
        dataset_clean["to"].append(len(tokens_list) - 1)
        dataset_clean["length"].append(len(tokens_list))
        dataset_clean["wrong_token"].append(s_tokens[0])
        dataset_clean["correct_token"].append(io_tokens[0])
        dataset_clean["S1_token"].append(s_tokens[0])
        dataset_clean["S2_token"].append(s_tokens[0])
        dataset_clean["IO_token"].append(io_tokens[0])
        dataset_clean["label"].append(io_tokens[0])
        dataset_clean["split"].append(types[i])

        a_tokens = model.to_str_tokens(" " + a_token)
        b_tokens = model.to_str_tokens(" " + b_token)
        c_tokens = model.to_str_tokens(" " + c_token)
        
        a_index = find_sublist_index(abc_tokens_list, a_tokens)
        if a_index == -1:
            print(f"A token '{a_token}' not found in tokens list for prompt: {abc_prompt}")
            print(f"Tokens list: {abc_tokens_list}, A token list: {a_tokens}")
            continue
        b_index = find_sublist_index(abc_tokens_list, b_tokens)
        if b_index == -1:
            print(f"B token '{b_token}' not found in tokens list for prompt: {abc_prompt}")
            print(f"Tokens list: {abc_tokens_list}, B token list: {b_tokens}")
            continue
        c_index = find_sublist_index(abc_tokens_list, c_tokens)
        if c_index == -1:
            print(f"C token '{c_token}' not found in tokens list for prompt: {abc_prompt}")
            print(f"Tokens list: {abc_tokens_list}, C token list: {c_tokens}")
            continue

        dataset_counter_abc["prompt"].append(abc_prompt)
        dataset_counter_abc["prompt_id"].append(template_index)
        dataset_counter_abc["prefix"].append(1)
        dataset_counter_abc["IO"].append(a_index)
        dataset_counter_abc["connector"].append(a_index + len(a_tokens))
        dataset_counter_abc["S1"].append(b_index)
        dataset_counter_abc["action1"].append(b_index + len(b_tokens))
        dataset_counter_abc["S2"].append(c_index)
        dataset_counter_abc["action2"].append(c_index + len(c_tokens))
        dataset_counter_abc["to"].append(len(tokens_list) - 1)
        dataset_counter_abc["length"].append(len(tokens_list))
        dataset_counter_abc["wrong_token"].append(s_tokens[0])
        dataset_counter_abc["correct_token"].append(io_tokens[0])
        dataset_counter_abc["S1_token"].append(s_tokens[0])
        dataset_counter_abc["S2_token"].append(s_tokens[0])
        dataset_counter_abc["IO_token"].append(io_tokens[0])
        dataset_counter_abc["label"].append(io_tokens[0])
        dataset_counter_abc["split"].append(types[i])

    dataset_clean = pd.DataFrame.from_dict(dataset_clean)
    print("data size:", dataset_clean.shape[0])
    dataset_clean = dataset_clean.drop_duplicates()

    dataset_counter_abc = pd.DataFrame.from_dict(dataset_counter_abc)
    dataset_counter_abc = dataset_counter_abc[dataset_counter_abc.index.isin(
        dataset_clean.index)]

    dataset_clean = dataset_clean.sample(
        frac=1, random_state=seed).reset_index(drop=True)
    dataset_counter_abc = dataset_counter_abc.sample(
        frac=1, random_state=seed).reset_index(drop=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset_clean.to_csv(os.path.join(save_dir, f'IOI_ABBA_tr_data_clean.csv'))
    dataset_counter_abc.to_csv(os.path.join(
        save_dir, f'IOI_ABBA_tr_data_counter_abc.csv'))

    eval_model_on_ioi(model_name, type_dataset="ABBA_tr",
                      save_dir=save_dir, batch_size=8)

def create_IOI_hi_dataset_ABBA(model_name: str, save_dir: str, seed: int = 42) -> None:
    """
    Generate IOI (Indirect Object Identification) hindi dataset in ABBA format.

    This function creates a dataset of sentences following the Indirect Object Identification (IOI)
    pattern in ABBA format, where names are arranged in an ABBA pattern (e.g., "Name2 Name1 Name1 Name2").
    The task is to identify the correct referent in sentences with this structure.

    Args:
        model_name (str): Name of the model being evaluated

    Returns:
        None: Saves generated datasets as CSV files in data/{model_name}/ioi/{seed}/ directories
    """

    HINDI_NAMES = [
        "अमित", "सुमित", "राहुल", "रोहित", "स्नेहा", 
        "पूजा", "नेहा", "अंजलि", "विकास", "समीर", 
        "रमेश", "सुरेश", "गीता", "सीता", "रवि", 
        "राजू", "नितिन", "आदित्य", "रिया", "प्रिया", 
        "दीपक", "करण", "अर्जुन", "मनोज", "संजय", 
        "किरण", "सोनिया", "पवन", "गगन", "आकाश", 
        "कमल", "विशाल", "अजय", "विजय", "सपना", 
        "आरती", "मीना", "टीना", "सुनीता", "अनीता", 
        "गौरव", "सौरभ", "आलोक", "वरुण", "तरुण", 
        "मोनिका", "शिखा", "निशा", "आशा", "उषा"
    ]

    PLACES = [
        "बाज़ार",      # Market
        "पार्क",       # Park
        "स्कूल",       # School
        "अस्पताल",     # Hospital
        "दुकान",       # Shop
        "मैदान",       # Field/Ground
        "मॉल",         # Mall
        "पुस्तकालय",   # Library
        "गाँव",        # Village
        "शहर"          # City
    ]
    OBJECTS = [
        "किताब",       # Book
        "कलम",         # Pen
        "गेंद",         # Ball
        "फोन",         # Phone
        "लैपटॉप",      # Laptop
        "बस्ता",       # Bag
        "घड़ी",        # Watch
        "खिलौना",      # Toy
        "छाता",        # Umbrella
        "चश्मा"        # Glasses
    ]

    ABBA_TEMPLATES = [
        "कल [A] और [B] [PLACE] गए थे। वहाँ एक [OBJECT] खरीदने के बाद, [B] ने",
        "सुबह [A] और [B] [PLACE] पहुँचे। अपना [OBJECT] निकालने के तुरंत बाद, [B] ने",
        "जब [A] और [B] [PLACE] में घूम रहे थे, तब वह [OBJECT] उठाकर [B] ने",
        "दोपहर को [A] और [B] [PLACE] में मिले। अपना [OBJECT] ढूँढने के बाद, [B] ने",
        "जैसे ही [A] और [B] [PLACE] के अंदर गए, वह [OBJECT] चुपके से लेकर [B] ने",
        "शाम को [A] और [B] [PLACE] से लौटे। रास्ते में वह [OBJECT] निकालकर [B] ने सीधे",
        "छुट्टियों में [A] और [B] [PLACE] गए थे। वहाँ एक सुंदर [OBJECT] पैक करने के बाद, [B] ने",
        "पिछली बार जब [A] और [B] [PLACE] में थे, तब वह [OBJECT] झट से निकालकर [B] ने",
        "जब [A] और [B] [PLACE] में बात कर रहे थे, तब अपना [OBJECT] पकड़कर [B] ने",
        "आज [A] और [B] [PLACE] पर रुके। अपना पुराना [OBJECT] बाहर लाने के बाद, [B] ने"
    ]

    ABC_TEMPLATES = [
        # Replace second [B] with [C]
        template[:template.find('[B]', template.find('[B]') + 1)] + '[C]' +
        template[template.find('[B]', template.find('[B]') + 1) + len('[B]'):]
        for template in ABBA_TEMPLATES
    ]

    ABBA_FULL_TEMPLATES = []
    ABC_FULL_TEMPLATES = []

    for template in ABBA_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                ABBA_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))
    for template in ABC_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                ABC_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))

    dtype = "bf16" if "Llama" else "float32"
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        # default_prepend_bos=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    names_comb = [c for c in itertools.combinations(HINDI_NAMES, 5)
                  # Require different start token for s1 and IO, and same length for clean and counterfactual
                  if c[0][0] != c[1][0] and
                  2*len(c[0]) + len(c[1]) == len(c[2]) + len(c[3]) + len(c[4])]
    dataset_size = 30000
    print("comb:", len(names_comb))
    random.seed(seed)
    types = random.choices(["circuit", "eval", "ablation"], weights=[
                           10, 10, 80], k=len(names_comb))
    dataset_clean = defaultdict(list)
    dataset_counter_abc = defaultdict(list)
    names_comb_seed = random.sample(names_comb, dataset_size)
    for i in tqdm(range(len(names_comb_seed))):
        s_token, io_token, a_token, b_token, c_token = names_comb_seed[i]
        template_index = random.randint(0, len(ABBA_FULL_TEMPLATES) - 1)
        baba_prompt = ABBA_FULL_TEMPLATES[template_index].replace(
            "[A]", io_token).replace("[B]", s_token)
        tokens_list = model.to_str_tokens(baba_prompt, prepend_bos=True)
        abc_prompt = ABC_FULL_TEMPLATES[template_index].replace(
            "[A]", a_token).replace("[B]", b_token).replace("[C]", c_token)
        abc_tokens_list = model.to_str_tokens(abc_prompt, prepend_bos=True)
        if len(tokens_list) != len(abc_tokens_list):
            continue

        io_tokens = model.to_str_tokens(" " + io_token)
        # io_tokens_ns = model.to_str_tokens(io_token)
        s_tokens = model.to_str_tokens(" " + s_token)
        # s_tokens_ns = model.to_str_tokens(s_token)
        io_index = find_sublist_index(tokens_list, io_tokens)
        if io_index == -1:
            print(f"IO token '{io_token}' not found in tokens list for prompt: {baba_prompt}")
            print(f"Tokens list: {tokens_list}, IO token list: {io_tokens}")
            continue
        s1_index = find_sublist_index(tokens_list, s_tokens)
        if s1_index == -1:
            print(f"S1 token '{s_token}' not found in tokens list for prompt: {baba_prompt}")
            print(f"Tokens list: {tokens_list}, S1 token list: {s_tokens}")
            continue
        s2_index = find_sublist_index(tokens_list[s1_index + 1:], s_tokens) + s1_index + 1
        dataset_clean["prompt"].append(baba_prompt)
        dataset_clean["prompt_id"].append(template_index)
        dataset_clean["prefix"].append(1)
        dataset_clean["IO"].append(io_index)
        dataset_clean["connector"].append(io_index + len(io_tokens))
        dataset_clean["S1"].append(s1_index)
        dataset_clean["action1"].append(s1_index + len(s_tokens))
        dataset_clean["S2"].append(s2_index)
        dataset_clean["action2"].append(s2_index + len(s_tokens))
        # Not really "to" in Turkish, but for consistency
        dataset_clean["to"].append(len(tokens_list) - 1)
        dataset_clean["length"].append(len(tokens_list))
        dataset_clean["wrong_token"].append(s_tokens[0])
        dataset_clean["correct_token"].append(io_tokens[0])
        dataset_clean["S1_token"].append(s_tokens[0])
        dataset_clean["S2_token"].append(s_tokens[0])
        dataset_clean["IO_token"].append(io_tokens[0])
        dataset_clean["label"].append(io_tokens[0])
        dataset_clean["split"].append(types[i])

        a_tokens = model.to_str_tokens(" " + a_token)
        b_tokens = model.to_str_tokens(" " + b_token)
        c_tokens = model.to_str_tokens(" " + c_token)
        
        a_index = find_sublist_index(abc_tokens_list, a_tokens)
        if a_index == -1:
            print(f"A token '{a_token}' not found in tokens list for prompt: {abc_prompt}")
            print(f"Tokens list: {abc_tokens_list}, A token list: {a_tokens}")
            continue
        b_index = find_sublist_index(abc_tokens_list, b_tokens)
        if b_index == -1:
            print(f"B token '{b_token}' not found in tokens list for prompt: {abc_prompt}")
            print(f"Tokens list: {abc_tokens_list}, B token list: {b_tokens}")
            continue
        c_index = find_sublist_index(abc_tokens_list, c_tokens)
        if c_index == -1:
            print(f"C token '{c_token}' not found in tokens list for prompt: {abc_prompt}")
            print(f"Tokens list: {abc_tokens_list}, C token list: {c_tokens}")
            continue

        dataset_counter_abc["prompt"].append(abc_prompt)
        dataset_counter_abc["prompt_id"].append(template_index)
        dataset_counter_abc["prefix"].append(1)
        dataset_counter_abc["IO"].append(a_index)
        dataset_counter_abc["connector"].append(a_index + len(a_tokens))
        dataset_counter_abc["S1"].append(b_index)
        dataset_counter_abc["action1"].append(b_index + len(b_tokens))
        dataset_counter_abc["S2"].append(c_index)
        dataset_counter_abc["action2"].append(c_index + len(c_tokens))
        dataset_counter_abc["to"].append(len(tokens_list) - 1)
        dataset_counter_abc["length"].append(len(tokens_list))
        dataset_counter_abc["wrong_token"].append(s_tokens[0])
        dataset_counter_abc["correct_token"].append(io_tokens[0])
        dataset_counter_abc["S1_token"].append(s_tokens[0])
        dataset_counter_abc["S2_token"].append(s_tokens[0])
        dataset_counter_abc["IO_token"].append(io_tokens[0])
        dataset_counter_abc["label"].append(io_tokens[0])
        dataset_counter_abc["split"].append(types[i])

    dataset_clean = pd.DataFrame.from_dict(dataset_clean)
    print("data size:", dataset_clean.shape[0])
    dataset_clean = dataset_clean.drop_duplicates()

    dataset_counter_abc = pd.DataFrame.from_dict(dataset_counter_abc)
    dataset_counter_abc = dataset_counter_abc[dataset_counter_abc.index.isin(
        dataset_clean.index)]

    dataset_clean = dataset_clean.sample(
        frac=1, random_state=seed).reset_index(drop=True)
    dataset_counter_abc = dataset_counter_abc.sample(
        frac=1, random_state=seed).reset_index(drop=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset_clean.to_csv(os.path.join(save_dir, f'IOI_ABBA_hi_data_clean.csv'))
    dataset_counter_abc.to_csv(os.path.join(
        save_dir, f'IOI_ABBA_hi_data_counter_abc.csv'))

    eval_model_on_ioi(model_name, type_dataset="ABBA_hi",
                      save_dir=save_dir, batch_size=8)


def create_IOI_es_dataset_ABBA(model_name: str, save_dir: str, seed: int = 42) -> None:
    """
    Generate IOI (Indirect Object Identification) spanish dataset in ABBA format.

    This function creates a dataset of sentences following the Indirect Object Identification (IOI) 
    pattern in ABBA format, where names are arranged in an ABBA pattern (e.g., "Name1 Name2 Name2 Name1").
    The task is to identify the correct referent in sentences with this structure.

    Args:
        model_name (str): Name of the model being evaluated

    Returns:
        None: Saves generated datasets as CSV files in data/{model_name}/ioi/{seed}/ directories

    """
    NAMES = ['Ana',
             'Sergio',
             'Lola',
             'Alan',
             'Oscar',
             'Laura',
             'Martin',
             'María',
             'Rosa',
             'Kelly',
             'Paula',
             'Eric',
             'Rafael',
             'Gabriel',
             'Jeffrey',
             'Carlos',
             'Dustin',
             'Kenneth',
             'Antonio',
             'Carmen',
             'Kevin',
             'Elena',
             'José',
             'Andrea',
             'Jaime',
             'Samuel',
             'Gloria',
             'Alejandro',
             'Cody',
             'Bruno',
             'Jorge',
             'Diego',
             'Victor',
             'Daniel',
             'Roberto',
             'Diana',
             'Isabel',
             'Miguel',
             'Juan',
             'Hugo',
             'Kimberly',
             'Kristen',
             'Felix',
             'Amanda',
             'Mario',
             'Travis',
             'Bradley',
             'Pablo',
             'Clara',
             'Alison',
             'Shannon',
             'Jonathan',
             'Samantha',
             'Sara',
             'Megan',
             'Justin',
             'Michelle',
             'Lindsay',
             'Bryan',
             'Jesse',
             'Erin',
             'Jennifer',
             'Manuel',
             'Eva',
             'Sean',
             'Vanessa',
             'Pedro',
             'Melissa',
             'Marcos',
             'Erica',
             'Brian',
             'Tiffany',
             'Luis',
             'Ricardo',
             'Lucas',
             'Max',
             'Julia',
             'Lisa',
             'Cristina',
             'Leo',
             'Nicole',
             'David',
             'Alicia',
             'Hector',
             'Joel',
             'Courtney',
             'Ivan']
    # Included 'el/la' and 'al' logic can be tricky,
    # so we include the preposition+article for the location to make templates simpler.
    PLACES = [
        "a la tienda",
        "al jardín",
        "al restaurante",
        "a la escuela",
        "al hospital",
        "a la oficina",
        "a la casa",
        "a la estación",
    ]

    # All objects selected are Masculine to ensure "dárselo" (give it) works for all.
    OBJECTS = [
        "un anillo",
        "un beso",
        "un hueso",
        "un balón",         # basketball
        "un ordenador",     # computer (using masculine form)
        "un collar",
        "un refresco",      # drink (using masculine form)
        "un bocadillo",     # snack (using masculine form)
    ]

    BABA_TEMPLATES = [
        "Entonces, [A] y [B] fueron [PLACE]. [B] le dio [OBJECT] a",
        "Entonces, [A] y [B] se divirtieron mucho en [PLACE]. [B] le dio [OBJECT] a",
        "Entonces, [A] y [B] estaban trabajando en [PLACE]. [B] decidió darle [OBJECT] a",
        "Entonces, [A] y [B] estaban pensando en ir [PLACE]. [B] quería darle [OBJECT] a",
        "Entonces, [A] y [B] tuvieron una larga discusión, y después [B] le dijo a",
        "Después de que [A] y [B] fueran [PLACE], [B] le dio [OBJECT] a",
        "Cuando [A] y [B] consiguieron [OBJECT] en [PLACE], [B] decidió dárselo a",
        "Cuando [A] y [B] consiguieron [OBJECT] en [PLACE], [B] decidió dar el objeto a",
        "Mientras [A] y [B] trabajaban en [PLACE], [B] le dio [OBJECT] a",
        "Mientras [A] y [B] iban [PLACE], [B] le dio [OBJECT] a",
        "Después del almuerzo, [A] y [B] fueron [PLACE]. [B] le dio [OBJECT] a",
        "Más tarde, [A] y [B] fueron [PLACE]. [B] le dio [OBJECT] a",
        "Entonces, [A] y [B] tuvieron una larga discusión. Después [B] le dijo a",
        "[PLACE] al que fueron [A] y [B] tenía [OBJECT]. [B] se lo dio a",
        "Los amigos [A] y [B] encontraron [OBJECT] en [PLACE]. [B] se lo dio a",
    ]

    ABC_TEMPLATES = [
        "Entonces, [B] y [A] fueron [PLACE]. [C] le dio [OBJECT] a",
        "Entonces, [B] y [A] se divirtieron mucho en [PLACE]. [C] le dio [OBJECT] a",
        "Entonces, [B] y [A] estaban trabajando en [PLACE]. [C] decidió darle [OBJECT] a",
        "Entonces, [B] y [A] estaban pensando en ir [PLACE]. [C] quería darle [OBJECT] a",
        "Entonces, [B] y [A] tuvieron una larga discusión, y después [C] le dijo a",
        "Después de que [B] y [A] fueran [PLACE], [C] le dio [OBJECT] a",
        "Cuando [B] y [A] consiguieron [OBJECT] en [PLACE], [C] decidió dárselo a",
        "Cuando [B] y [A] consiguieron [OBJECT] en [PLACE], [C] decidió dar el objeto a",
        "Mientras [B] y [A] trabajaban en [PLACE], [C] le dio [OBJECT] a",
        "Mientras [B] y [A] iban [PLACE], [C] le dio [OBJECT] a",
        "Después del almuerzo, [B] y [A] fueron [PLACE]. [C] le dio [OBJECT] a",
        "Más tarde, [B] y [A] fueron [PLACE]. [C] le dio [OBJECT] a",
        "Entonces, [B] y [A] tuvieron una larga discusión. Después [C] le dijo a",
        "[PLACE] al que fueron [B] y [A] tenía [OBJECT]. [C] se lo dio a",
        "Los amigos [B] y [A] encontraron [OBJECT] en [PLACE]. [C] se lo dio a",
    ]

    BABA_FULL_TEMPLATES = []
    ABC_FULL_TEMPLATES = []

    for template in BABA_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                BABA_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))
    for template in ABC_TEMPLATES:
        for place in PLACES:
            for obj in OBJECTS:
                ABC_FULL_TEMPLATES.append(template.replace(
                    "[PLACE]", place).replace("[OBJECT]", obj))

    dtype = "bf16" if "Llama" else "float32"
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    # name_token_count = {}
    # for name in NAMES:
    #     name_token_count[name] = len(
    #         model.to_str_tokens(" " + name)
    #     )

    names_comb = list(itertools.combinations(NAMES, 5))
    dataset_size = 30000
    print("comb:", len(names_comb))
    random.seed(seed)
    types = random.choices(["circuit", "eval", "ablation"], weights=[
                           10, 10, 80], k=len(names_comb))
    dataset_clean = defaultdict(list)
    dataset_counter_abc = defaultdict(list)
    names_comb_seed = random.sample(names_comb, dataset_size)
    for i in tqdm(range(len(names_comb_seed))):
        s_token, io_token, a_token, b_token, c_token = names_comb_seed[i]
        template_index = random.randint(0, len(BABA_FULL_TEMPLATES) - 1)
        baba_prompt = BABA_FULL_TEMPLATES[template_index].replace(
            "[A]", io_token).replace("[B]", s_token)
        tokens_list = model.to_str_tokens(baba_prompt, prepend_bos=True)
        io_index = tokens_list.index(" " + io_token)
        s1_index = tokens_list.index(" " + s_token)
        s2_index = tokens_list[s1_index +
                               1:].index(" " + s_token) + s1_index + 1
        dataset_clean["prompt"].append(baba_prompt)
        dataset_clean["prompt_id"].append(template_index)
        dataset_clean["prefix"].append(1)
        dataset_clean["IO"].append(io_index)
        dataset_clean["and"].append(io_index + 1)
        dataset_clean["S1"].append(s1_index)
        dataset_clean["S1+1"].append(s1_index+1)
        dataset_clean["action1"].append(s1_index + 2)
        dataset_clean["S2"].append(s2_index)
        dataset_clean["action2"].append(s2_index + 1)
        dataset_clean["to"].append(len(tokens_list) - 1)
        dataset_clean["length"].append(len(tokens_list))
        dataset_clean["wrong_token"].append(" " + s_token)
        dataset_clean["correct_token"].append(" " + io_token)
        dataset_clean["S1_token"].append(" " + s_token)
        dataset_clean["S2_token"].append(" " + s_token)
        dataset_clean["IO_token"].append(" " + io_token)
        dataset_clean["label"].append(" " + io_token)
        dataset_clean["split"].append(types[i])

        abc_prompt = ABC_FULL_TEMPLATES[template_index].replace(
            "[A]", a_token).replace("[B]", b_token).replace("[C]", c_token)

        dataset_counter_abc["prompt"].append(abc_prompt)
        dataset_counter_abc["prompt_id"].append(template_index)
        dataset_counter_abc["prefix"].append(1)
        dataset_counter_abc["IO"].append(io_index)
        dataset_counter_abc["and"].append(io_index + 1)
        dataset_counter_abc["S1"].append(s1_index)
        dataset_counter_abc["S1+1"].append(s1_index+1)
        dataset_counter_abc["action1"].append(s1_index + 2)
        dataset_counter_abc["S2"].append(s2_index)
        dataset_counter_abc["action2"].append(s2_index + 1)
        dataset_counter_abc["to"].append(len(tokens_list) - 1)
        dataset_counter_abc["length"].append(len(tokens_list))
        dataset_counter_abc["wrong_token"].append(" " + s_token)
        dataset_counter_abc["correct_token"].append(" " + io_token)
        dataset_counter_abc["S1_token"].append(" " + s_token)
        dataset_counter_abc["S2_token"].append(" " + s_token)
        dataset_counter_abc["IO_token"].append(" " + io_token)
        dataset_counter_abc["label"].append(" " + io_token)
        dataset_counter_abc["split"].append(types[i])

    dataset_clean = pd.DataFrame.from_dict(dataset_clean)
    print("data size:", dataset_clean.shape[0])
    dataset_clean = dataset_clean.drop_duplicates()

    dataset_counter_abc = pd.DataFrame.from_dict(dataset_counter_abc)
    dataset_counter_abc = dataset_counter_abc[dataset_counter_abc.index.isin(
        dataset_clean.index)]

    dataset_clean = dataset_clean.sample(
        frac=1, random_state=seed).reset_index(drop=True)
    dataset_counter_abc = dataset_counter_abc.sample(
        frac=1, random_state=seed).reset_index(drop=True)

    dataset_clean.to_csv(os.path.join(save_dir, f'IOI_ABBA_data_clean.csv'))
    dataset_counter_abc.to_csv(os.path.join(
        save_dir, f'IOI_ABBA_data_counter_abc.csv'))

    eval_model_on_ioi(model_name=model_name,
                      type_dataset="ABBA", save_dir=save_dir)


def is_name_combination_valid(name_comb: Tuple[str, str, str, str, str], names_token_count) -> bool:
    # Ensure s1 and IO start with different letters
    if name_comb[0][0] == name_comb[1][0]:
        return False
    clean_len = 2*names_token_count[name_comb[0]
                                    ] + names_token_count[name_comb[1]]
    cf_len = names_token_count[name_comb[2]] + \
        names_token_count[name_comb[3]] + names_token_count[name_comb[4]]
    if clean_len != cf_len:
        return False
    return True


def eval_model_on_ioi(model_name: str, type_dataset: str, save_dir: str, batch_size: int = 8) -> None:
    """
    Evaluate a model's performance on the IOI (Indirect Object Identification) task.

    This function evaluates a model on the IOI task by processing datasets containing prompts in either 
    ABBA or BABA format. It computes probabilities for the model's predictions and saves evaluation metrics
    including top answers and their probabilities.

    Args:
        model (str): Name/path of the model to evaluate
        type_dataset (str): Format of the IOI dataset ("ABBA" or "BABA")
        batch_size (int, optional): Batch size for processing. Defaults to 8.

    Returns:
        None: Results are saved back to the original CSV files with additional columns:
            - top_answer: Model's predicted token
            - top_answer_prob: Probability of the predicted token
            - correct_prob: Probability of the correct token
            - wrong_prob: Probability of the incorrect token
    """

    files_list = []

    files_list = [
        os.path.join(save_dir, f"IOI_{type_dataset}_data_clean.csv"),
        os.path.join(save_dir, f"IOI_{type_dataset}_data_counter_abc.csv"),
    ]

    dtype = "bf16" if "Llama" else "float32"
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        default_prepend_bos=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )

    model.eval()
    for file in files_list:

        top_answer_list = []
        top_answer_prob_list = []
        correct_prob_list = []
        wrong_prob_list = []
        data = pd.read_csv(file, index_col=[0])
        num_batches = math.ceil(data.shape[0] / batch_size)
        print(file)
        for b in tqdm(range(num_batches)):
            batch = data.iloc[b *
                              batch_size: (b + 1) * batch_size].reset_index()
            logits = model(batch["prompt"].to_list(), return_type="logits")
            for index, row in batch.iterrows():
                outputs_token = torch.argmax(
                    logits[index, row["length"]-1], dim=-1)

                probs = torch.softmax(logits[index, row["length"]-1], dim=-1)
                output_prob = probs[outputs_token].item()
                correct_prob = probs[model.to_single_token(
                    row["correct_token"])].item()
                wrong_prob = probs[model.to_single_token(
                    row["wrong_token"])].item()

                top_answer_list.append(model.to_string(outputs_token))
                top_answer_prob_list.append(round(output_prob, 4))
                correct_prob_list.append(round(correct_prob, 4))
                wrong_prob_list.append(round(wrong_prob, 4))
            del batch, logits
        data = data.assign(top_answer=top_answer_list, top_answer_prob=top_answer_prob_list, correct_prob=correct_prob_list,
                           wrong_prob=wrong_prob_list)

        num_correct = data[data['top_answer']
                           == data['correct_token']].shape[0]
        num_wrong = data[data['top_answer'] == data['wrong_token']].shape[0]
        print("correct:", num_correct / data.shape[0])
        print("wrong:", num_wrong / data.shape[0])

        data.to_csv(file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Create datasets for model evaluation')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to evaluate')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save the generated datasets')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--task', type=str, required=True, choices=['ioi_baba', 'ioi_abba', 'wino_bias', 'greater_than'],
                        help='Task to generate dataset for')

    args = parser.parse_args()

    # Create directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    if args.task == 'ioi_baba':
        create_IOI_dataset_BABA(
            model_name=args.model_name,
            save_dir=args.save_dir,
            seed=args.seed
        )
    elif args.task == 'ioi_abba':
        create_IOI_dataset_ABBA(
            model_name=args.model_name,
            save_dir=args.save_dir,
            seed=args.seed
        )
    elif args.task == 'wino_bias':
        create_wino_bias_dataset(
            model_name=args.model_name,
            save_dir=args.save_dir,
            seed=args.seed
        )
    elif args.task == 'greater_than':
        create_greather_than_dataset(
            model_name=args.model_name,
            save_dir=args.save_dir,
            seed=args.seed
        )
