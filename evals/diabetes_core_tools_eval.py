"""Reference eval using the `dfkit.evaluation` harness against core tools only.

This module demonstrates the standard pattern for writing evals with the dfkit
evaluation harness. It uses the scikit-learn diabetes dataset and evaluates how
well an LLM agent uses the core DataFrame tools to answer analytical questions.

Expected pass-rate range: 0.7-1.0. LLM variance is normal; results below 0.7
on a given run do not necessarily indicate a regression.

Agent/judge split:
- **Agent (Haiku)**: Executes tools and produces answers.
- **Judge (Sonnet)**: Extracts structured facts from the agent's free-text
  response and compares them against the expected `EvalCase` ground truth.

How to run::

    uv run python evals/diabetes_core_tools_eval.py

To add verbose tool-call logging, edit the ``__main__`` guard to::

    with enable_logging(): main()
"""

from __future__ import annotations

import sys
from typing import Final, Literal

import polars as pl
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel
from sklearn import datasets

from dfkit import DataFrameToolkit
from dfkit.chat_models_config import ModelName, get_chat_model
from dfkit.evaluation import (
    Confounders,
    Correlation,
    EvalCase,
    EvalResult,
    EvalSummary,
    Metric,
    NonNegativeMetric,
    RelationshipStrength,
    SampleCount,
    evaluate_agent,
    summarize_results,
)

RTOL: Final[float] = 0.1
ATOL: Final[float] = 0.05
CONFOUNDER_CORR_THRESHOLD: Final[float] = 0.2
PASS_RATE_THRESHOLD: Final[float] = 0.5
DATASET_NAME: Final[str] = "Diabetes Progression Dataset"


# region Load dataset


def load_diabetes_dataset() -> pl.DataFrame:
    """Load the scikit-learn diabetes dataset as a polars DataFrame.

    Mirrors the construction in `examples/diabetes_dataset_exploration.ipynb`:
    loads the unscaled diabetes data, maps the numeric `sex` column to
    `"male"`/`"female"` strings, and appends the `disease_progression` target.

    Returns:
        pl.DataFrame: Diabetes dataset with 11 columns and 442 rows.
    """
    data, target = datasets.load_diabetes(return_X_y=True, scaled=False)
    df = pl.DataFrame(
        data=data,
        schema=["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
    )
    return df.with_columns(
        pl.col("sex").map_elements(lambda x: "male" if x == 1 else "female", return_dtype=pl.String),
        pl.Series(target).alias("disease_progression"),
    )


def register_dataset(toolkit: DataFrameToolkit, df: pl.DataFrame) -> None:
    """Register the diabetes dataset with the toolkit.

    Calls `toolkit.register_dataframe` with the canonical name, description,
    and column descriptions from `examples/diabetes_dataset_exploration.ipynb`.

    Args:
        toolkit (DataFrameToolkit): The toolkit instance to register the dataset with.
        df (pl.DataFrame): The diabetes DataFrame returned by `load_diabetes_dataset`.
    """
    toolkit.register_dataframe(
        name=DATASET_NAME,
        dataframe=df,
        description="""
    Ten baseline variables, age, sex, body mass index, average blood pressure,
    and six blood serum measurements were obtained for each diabetes patient,
    as well as the response of interest, a quantitative measure of disease
    progression one year after baseline.
    """,
        column_descriptions={
            "age": "Age of the patient in years.",
            "sex": "Sex of the patient",
            "bmi": "Body mass index.",
            "bp": "Average blood pressure.",
            "s1": "TC, total serum cholesterol.",
            "s2": "LDL, low-density lipoproteins.",
            "s3": "HDL, high-density lipoproteins.",
            "s4": "TCH, total cholesterol / HDL.",
            "s5": "LTG, possibly log of serum triglycerides level.",
            "s6": "GLU, blood sugar level.",
            "disease_progression": "A quantitative measure of disease progression one year after baseline.",
        },
    )


# endregion
