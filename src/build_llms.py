import os
import numpy as np
import json
import random
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from typing import List

def get_seed_list() -> List[int]:
    """Read seeds from ../assets/seed_list.txt (one integer per line)."""
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        return [int(line.rstrip('\n')) for line in f]


# Load API key from disk (expects a file named 'api_key.txt' in the same directory)
def load_api_key(#
        path: str = "../keys/openai_api_key.txt"
) -> str:
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(f"API key file not found at: {path}")
    return key_path.read_text().strip()


def make_context(respondent: str, target: str) -> str:
    """
    respondent: 'rep' or 'dem'
    target:     'rep' or 'dem'
    """
    if respondent == "rep":
        who = (
            "ideologically conservative. "
            "Politically, they are a strong Republican. "
            "Racially, they are white. "
            "They are male. "
            "Financially, they are upper-middle class. "
            "In terms of age, they are young."
        )
    elif respondent == "dem":
        who = (
            " ideologically liberal. "
            "Politically, they are a strong Democrat. "
            "Racially, they are white. "
            "They are female. "
            "Financially, they are poor. "
            "In terms of age, they are old."
        )
    else:
        raise ValueError(f"Unknown respondent {respondent!r}")

    if target == "rep":
        target_str = "Republican"
    elif target == "dem":
        target_str = "Democratic"
    else:
        raise ValueError(f"Unknown target {target!r}")

    return (
        f"The respondent is {who}"
        f"They list words describing {target_str} voters. "
        "They give exactly the following words."
    )


def generate_labels_for_all_pairs(
    client,
    model_name: str,
    system_prompt: str,
    allowed_words_dem: List[str],
    allowed_words_rep: List[str],
    temperature: float,
    top_p: float,
    seed: int,
    rng: random.Random,
    shuffle_lists: bool = True,
) -> List[str]:
    """
    Helper to query all four respondent/target combinations.
    When shuffle_lists is False the provided word order is preserved.
    """

    def maybe_shuffle(words: List[str]) -> List[str]:
        words_copy = words[:]
        if shuffle_lists:
            rng.shuffle(words_copy)
        return words_copy

    def ask_model(respondent: str, target: str, allowed_words: List[str]) -> str:
        word_list = maybe_shuffle(allowed_words)
        ctx = make_context(respondent, target)
        prompt = f"{ctx}\n\nWords: {', '.join(word_list)}"

        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            frequency_penalty=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        try:
            label = json.loads(resp.choices[0].message.content)["label"]
        except Exception:
            label = allowed_words[0]
        if label not in allowed_words:
            label = allowed_words[0]
        return label

    return [
        ask_model("rep", "dem", allowed_words_dem),
        ask_model("dem", "rep", allowed_words_rep),
        ask_model("dem", "dem", allowed_words_dem),
        ask_model("rep", "rep", allowed_words_rep),
    ]


def main():
    import csv
    import json
    import random
    from tqdm import tqdm

    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    model_name = "gpt-4o-mini-2024-07-18"

    # ---------------------------------------------------------
    # stereotype lists
    # ---------------------------------------------------------

    # Republican-provided list describing Democrats
    allowed_words_dem = [
        "liberal", "young", "progressive", "poor", "open-minded", "female",
        "educated", "intelligent", "pro-choice", "compassionate", "caring",
        "socialist", "communist", "athiest",
    ]

    # Democrat-provided list describing Republicans
    allowed_words_rep = [
        "conservative", "white", "religious", "racist", "rich",
        "old", "christian", "male", "homophobic", "patriotic",
        "ignorant", "traditional", "selfish", "responsible",
    ]

    # ---------------------------------------------------------
    # system prompt enforcing 1-word JSON output
    # ---------------------------------------------------------

    system_prompt = (
        "You are given a list of words that a respondent uses to describe "
        "supporters of a political group.\n\n"
        "Your task:\n"
        "1. Choose exactly ONE word from that list.\n"
        "2. The chosen word MUST be exactly one of the words from that list.\n"
        "3. Output exactly a JSON object of the form {\"label\": \"<word>\"}.\n"
        "4. Output NOTHING else."
    )

    # ---------------------------------------------------------
    # sampling parameters
    # ---------------------------------------------------------

    temperatures = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    top_p = 0.95
    seeds = get_seed_list()[0:1000]

    rows = []

    for temperature in tqdm(temperatures, desc="temperatures"):
        for seed in tqdm(seeds, desc="seeds", leave=False):
            rng = random.Random(seed)

            label_1, label_2, label_3, label_4 = generate_labels_for_all_pairs(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                allowed_words_dem=allowed_words_dem,
                allowed_words_rep=allowed_words_rep,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                rng=rng,
                shuffle_lists=True,
            )

            # ===============================================================
            # store compact row (no raw JSON, no prompt_order)
            # ===============================================================

            rows.append({
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,

                "rep_to_dem_label": label_1,
                "dem_to_rep_label": label_2,
                "dem_to_dem_label": label_3,
                "rep_to_rep_label": label_4,
            })

    # ---------------------------------------------------------
    # write CSV
    # ---------------------------------------------------------

    if rows:
        fieldnames = list(rows[0].keys())
        with open("results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ----------------------------------------------------------------------
# NEW FUNCTION: 1000 calls with SAME seed and temperature = 0,
# saved to a separate CSV.
# ----------------------------------------------------------------------
def run_fixed_seed_temp0_experiment():
    import csv
    import json
    import random

    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    model_name = "gpt-4o-mini-2024-07-18"

    # stereotype lists (same as in main)
    allowed_words_dem = [
        "liberal", "young", "progressive", "poor", "open-minded", "female",
        "educated", "intelligent", "pro-choice", "compassionate", "caring",
        "socialist", "communist", "athiest",
    ]

    allowed_words_rep = [
        "conservative", "white", "religious", "racist", "rich",
        "old", "christian", "male", "homophobic", "patriotic",
        "ignorant", "traditional", "selfish", "responsible",
    ]

    system_prompt = (
        "You are given a list of words that a respondent uses to describe "
        "supporters of a political group.\n\n"
        "Your task:\n"
        "1. Choose exactly ONE word from that list.\n"
        "2. The chosen word MUST be exactly one of the words from that list.\n"
        "3. Output exactly a JSON object of the form {\"label\": \"<word>\"}.\n"
        "4. Output NOTHING else."
    )

    temperature = 0.0
    top_p = 0.95
    runs = 1000

    # use a single fixed seed for all 1000 calls
    seed = get_seed_list()[0]

    rows = []

    for run_id in range(runs):
        rng = random.Random(seed)

        label_1, label_2, label_3, label_4 = generate_labels_for_all_pairs(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            allowed_words_dem=allowed_words_dem,
            allowed_words_rep=allowed_words_rep,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            rng=rng,
            shuffle_lists=True,
        )

        rows.append({
            "run_id": run_id,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "rep_to_dem_label": label_1,
            "dem_to_rep_label": label_2,
            "dem_to_dem_label": label_3,
            "rep_to_rep_label": label_4,
        })

    if rows:
        fieldnames = list(rows[0].keys())
        with open("results_fixed_seed_temp0.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def run_fixed_seed_temp0_no_shuffle_experiment():
    """
    Same setup as run_fixed_seed_temp0_experiment but preserves the
    original word order (no list shuffling). Results are written to
    results_fixed_seed_temp0_no_shuffle.csv.
    """
    import csv
    import json
    import random

    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    model_name = "gpt-4o-mini-2024-07-18"

    allowed_words_dem = [
        "liberal", "young", "progressive", "poor", "open-minded", "female",
        "educated", "intelligent", "pro-choice", "compassionate", "caring",
        "socialist", "communist", "athiest",
    ]

    allowed_words_rep = [
        "conservative", "white", "religious", "racist", "rich",
        "old", "christian", "male", "homophobic", "patriotic",
        "ignorant", "traditional", "selfish", "responsible",
    ]

    system_prompt = (
        "You are given a list of words that a respondent uses to describe "
        "supporters of a political group.\n\n"
        "Your task:\n"
        "1. Choose exactly ONE word from that list.\n"
        "2. The chosen word MUST be exactly one of the words from that list.\n"
        "3. Output exactly a JSON object of the form {\"label\": \"<word>\"}.\n"
        "4. Output NOTHING else."
    )

    temperature = 0.0
    top_p = 0.95
    runs = 1000
    seed = get_seed_list()[0]
    rows = []

    for run_id in range(runs):
        rng = random.Random(seed)
        label_1, label_2, label_3, label_4 = generate_labels_for_all_pairs(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            allowed_words_dem=allowed_words_dem,
            allowed_words_rep=allowed_words_rep,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            rng=rng,
            shuffle_lists=False,
        )

        rows.append({
            "run_id": run_id,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "rep_to_dem_label": label_1,
            "dem_to_rep_label": label_2,
            "dem_to_dem_label": label_3,
            "rep_to_rep_label": label_4,
        })

    if rows:
        fieldnames = list(rows[0].keys())
        with open("results_fixed_seed_temp0_no_shuffle.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
    run_fixed_seed_temp0_experiment()
    run_fixed_seed_temp0_no_shuffle_experiment()
