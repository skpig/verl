import json
try:
    from verifiable_tasks import *
    registried_tasks = {
        "FOLIO": FOLIO_verify,
        # "SPARTUN": SPARTUN_verify,
        # "brick_world": brick_world_verify,
        "crosswords_puzzle": crosswords_puzzle_verify,
        "crypto_puzzle": crypto_puzzle_verify,
        "detectbench": detectbench_verify,
        "fifteenpuzzle": fifteenpuzzle_verify,
        "game24": game24_verify,
        "grid_puzzle": grid_puzzle_verify,
        "knights_and_knaves": knights_and_knaves_verify,
        "magic_square": magic_square_verify,
        "maze": maze_verify,
        "natural_language_navigation": natural_language_navigation_verify,
        # "nlvr_based_manipulation": nlvr_based_manipulation_verify,
        "sixteenpuzzle": sixteenpuzzle_verify,
        "skyscraper": skyscraper_verify,
        "stack_permutation": stack_permutation_verify,
        "sudoku": sudoku_verify,
        "sudoku_2x2": sudoku_2x2_verify,
        "sum_skyscraper": sum_skyscraper_verify,
        # "true_detect": true_detect_verify,
        "twiddle": twiddle_verify,
        "zebra_logic": zebra_logic_verify,
    }
except Exception as ex:
    registried_tasks = {}

    def dummy_verify(solution_str, answer, meta):
        return 0


def compute_score(solution_str, ground_truth, **argv) -> float:
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    meta = ground_truth["meta"]
    answer = ground_truth["answer"]
    task_name = ground_truth["task_name"]
    try:
        verify_fn = registried_tasks[task_name]
    except Exception as ex:
        return dummy_verify
    return verify_fn(solution_str, answer, meta)
