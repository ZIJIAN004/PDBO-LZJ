"""Problem builders and parsers for PDBO."""

from problem_parser import (
    evaluate_LABS_bits,
    generate_LABS,
    generate_Max_cut,
    generate_MIS,
    generate_max_sat,
    parse_gset,
    random_graph,
)

generate_labs = generate_LABS
generate_max_cut = generate_Max_cut
generate_mis = generate_MIS
evaluate_labs_bits = evaluate_LABS_bits

__all__ = [
    "evaluate_labs_bits",
    "generate_labs",
    "generate_max_cut",
    "generate_max_sat",
    "generate_mis",
    "parse_gset",
    "random_graph",
]
