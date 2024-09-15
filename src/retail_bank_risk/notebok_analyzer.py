import json
import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor


def analyze_and_reduce_notebook(notebook_path, output_path=None):
    """
    Analyze a Jupyter notebook for size issues and optionally create a reduced version.

    :param notebook_path: Path to the input notebook
    :param output_path: Path to save the reduced notebook (optional)
    :return: Dict with analysis results
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    total_size = len(json.dumps(nb))
    cell_sizes = []
    output_sizes = []

    for cell in nb["cells"]:
        cell_size = len(json.dumps(cell))
        cell_sizes.append(cell_size)

        if "outputs" in cell:
            output_size = sum(
                len(json.dumps(output)) for output in cell["outputs"]
            )
            output_sizes.append(output_size)

    analysis = {
        "total_size_mb": total_size / (1024 * 1024),
        "num_cells": len(cell_sizes),
        "avg_cell_size_kb": (
            sum(cell_sizes) / len(cell_sizes) / 1024 if cell_sizes else 0
        ),
        "max_cell_size_mb": (
            max(cell_sizes) / (1024 * 1024) if cell_sizes else 0
        ),
        "total_output_size_mb": (
            sum(output_sizes) / (1024 * 1024) if output_sizes else 0
        ),
    }

    if output_path:
        # Clear all outputs
        nb = nbformat.reads(json.dumps(nb), as_version=4)
        ClearOutputPreprocessor().preprocess(nb, {})

        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        analysis["reduced_size_mb"] = len(json.dumps(nb)) / (1024 * 1024)

    return analysis
