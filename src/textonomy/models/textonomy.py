import json
import logging
import random
import time
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from IPython.display import display
from mlflow.models import infer_signature
import instructor
from instructor import from_mistral
import litellm
import openai as openai_og
from openai import OpenAI
from mistralai import Mistral
from litellm import completion
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    classification_report,
    completeness_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Pydantic models for the LLM response
# ------------------------------------------------------------------------------


class SummaryExtraction(BaseModel):
    explanation: str = Field(..., description="Your answer to Q1.")
    summary: str = Field(..., description="Your answer to Q2.")


class Label(BaseModel):
    id: int
    description: str
    name: str


class Taxonomy(BaseModel):
    explanation: str
    rating: int
    edits: str
    labels: List[Label]

    def to_pretty_json(self) -> str:
        return json.dumps(self.model_dump(), indent=4)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert labels to a Pandas DataFrame for easy display."""
        return pd.DataFrame([label.model_dump() for label in self.labels])

    def print_with_dataframe(self) -> None:
        """Print the taxonomy explanation and edits, and display the labels as a DataFrame."""
        print("Score for current taxonomy: ", self.rating)
        print("Explanation: ", self.explanation)
        print("Suggested edits: ", self.edits)
        print("Taxonomy: ")
        display(self.to_dataframe())


class DocClassification(BaseModel):
    explanation: str = Field(..., description="Your explanation for the label assignment.")
    id: int = Field(..., description="Your chosen label id from the reference table or -1")
    name: str = Field(..., description="Your chosen label name from the reference table or 'Undefined'")


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def call_llm(
    self,
    messages: List[Dict[str, str]],
    response_format: Any,
    model: str = "gpt-4o-mini-2024-07-18",
    temperature: float = 0.1,
    max_tokens: int = 1048,
    retries: int = 10,
    wait_time: int = 5,
) -> Any:
    """
    Wrapper around the litellm.completion API for structured output.
    messages should be a list of dicts with 'role' and 'content'.
    response_format should be a Pydantic model for the response.
    """
    if self.debug:
        litellm._turn_on_debug()
    else:
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    if "openai/" in model:
        model = model.replace("openai/", "")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        for attempt in range(retries):
            try:
                if model == "o3-mini-2025-01-31":
                    response = client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        max_completion_tokens=max_tokens,
                    )
                else:
                    response = client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    )
                parsed_response = response.choices[0].message.parsed
                return parsed_response
            except openai_og.RateLimitError as e:
                if attempt < retries - 1:
                    if self.debug:
                        print(f"Rate limit exceeded: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Maximum retries reached. Raising the error.")
                    raise
    elif "mistral/" in model:
        model = model.replace("mistral/", "")
        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

        for attempt in range(retries):
            try:
                #client = instructor.from_litellm(completion, mode=instructor.Mode.TOOLS) # , mode=instructor.Mode.JSON
                instructor_client = instructor.from_mistral(client=client, model=model, mode=instructor.Mode.MISTRAL_STRUCTURED_OUTPUTS)
                parsed_response = instructor_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_model=response_format,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=1,
                    strict=True,
                )
                return parsed_response
            except litellm.RateLimitError as e: # MODIFIED: Import exception from litellm
                if attempt < retries - 1:
                    if self.debug:
                        print(f"Rate limit exceeded: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Maximum retries reached. Raising the error.")
                    raise
            except Exception as e:
                if attempt < retries - 1:
                    if self.debug:
                        print(f"Error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Maximum retries reached. Raising the error.")
                    raise
    
    


def calculate_purity(y_true, y_pred) -> tuple[float, float, float]:
    """
    Calculate harmonic purity between two set of clusterings

    Parameters:
    - true_col: Column containing a ground-truth label for each document
    - pred_col: Column containing a predicted label for each document (containing parsed topics)
    - df: Pandas data frame containing two columns (true_col and pred_col)

    Returns:
    - purity: Purity score
    - inverse_purity: Inverse purity score
    - harmonic_purity: Harmonic purity score

    Adapated this original implementation: Git Repo for TopicGPT by Pham et al. (2024: http://arxiv.org/abs/2311.01449)
    """
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true=y_true, labels_pred=y_pred)
    col_sums = contingency_matrix.sum(axis=0, keepdims=True)  # type: ignore
    row_sums = contingency_matrix.sum(axis=1, keepdims=True)  # type: ignore

    precision = np.divide(
        contingency_matrix,
        col_sums,
        out=np.zeros_like(contingency_matrix, dtype=float),
        where=col_sums != 0,
    )
    recall = np.divide(
        contingency_matrix,
        row_sums,
        out=np.zeros_like(contingency_matrix, dtype=float),
        where=row_sums != 0,
    )

    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(precision),
        where=denom != 0,
    )
    purity = (
        np.amax(precision, axis=0) * contingency_matrix.sum(axis=0)  # type: ignore
    ).sum() / contingency_matrix.sum()  # type: ignore
    inverse_purity = (
        np.amax(recall, axis=1) * contingency_matrix.sum(axis=1)  # type: ignore
    ).sum() / contingency_matrix.sum()  # type: ignore
    harmonic_purity = (
        np.amax(f1, axis=1) * contingency_matrix.sum(axis=1)  # type: ignore
    ).sum() / contingency_matrix.sum()  # type: ignore
    return (purity, inverse_purity, harmonic_purity)


# ------------------------------------------------------------------------------
# Textonomy model class
# ------------------------------------------------------------------------------


class Textonomy:
    """
    A class to generate and refine taxonomies from a corpus of texts using LLMs via litellm.
    """

    def __init__(
        self,
        use_case: str = "Topic modeling",
        data_description: str = "Documents from a big corpus of documents",
        temperature: float = 0.1,
        max_completion_tokens=1000,
        retries: int = 5,
        wait_time: int = 10,
        summaries_max_thread_workers: int = 20,
        cluster_name_length: int = 4,
        cluster_description_length: int = 15,
        max_num_clusters: int = 50,
        summary_length: int = 20,
        summary_explanation_length: int = 30,
        taxonomy_explanation_length: int = 50,
        classification_explanation_length: int = 15,
        taxonomy_update_suggestion_length: int = 150,
        taxonomy_refinement_iterations: int = 4,  # including the initial and review steps (minimum 2)
        summary_model: str = "openai/gpt-4o-mini-2024-07-18",
        taxonomy_generation_model: str = "openai/gpt-4o-2024-08-06",
        synthesis_mode: bool = True,
        llm_classification_model: str = "openai/gpt-4o-mini-2024-07-18",
        classifier_model: Literal["LogisticRegression", "XGBoost"] = "LogisticRegression",
        encoder_model_name: str = "all-MiniLM-L6-v2",
        taxonomy_sample_ratio: float = 0.05,
        classifier_sample_ratio: float = 0.1,
        classifier_train_test_split_ratio: float = 0.75,
        shuffle_training_data: bool = True,
        shuffle_taxonomy_data: bool = True,
        shuffle_classifer_data: bool = True,
        shuffle_random_state=None,
        verbose: bool = True,
        debug: bool = False,
    ):
        self.use_case = use_case
        self.data_description = data_description
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.retries = retries
        self.wait_time = wait_time
        self.summaries_max_thread_workers = summaries_max_thread_workers
        self.cluster_name_length = cluster_name_length  # in words
        self.cluster_description_length = cluster_description_length  # in words
        self.max_num_clusters = max_num_clusters
        self.summary_length = summary_length  # in words
        self.summary_explanation_length = summary_explanation_length  # in words
        self.taxonomy_explanation_length = taxonomy_explanation_length  # in words
        self.classification_explanation_length = classification_explanation_length  # in words
        self.taxonomy_update_suggestion_length = taxonomy_update_suggestion_length
        self.taxonomy_refinement_iterations = (
            taxonomy_refinement_iterations  # including the initial and review steps (minimum 2)
        )
        self.num_batches = taxonomy_refinement_iterations - 1
        self.summary_model = summary_model
        self.taxonomy_generation_model = taxonomy_generation_model
        self.synthesis_mode = synthesis_mode
        self.llm_classification_model = llm_classification_model
        self.classifier_model = classifier_model
        self.encoder_model_name = encoder_model_name
        self.taxonomy_sample_ratio = taxonomy_sample_ratio
        self.classifier_sample_ratio = classifier_sample_ratio
        self.classifier_train_test_split_ratio = classifier_train_test_split_ratio
        self.shuffle_training_data = shuffle_training_data
        self.shuffle_taxonomy_data = shuffle_taxonomy_data
        self.shuffle_classifer_data = shuffle_classifer_data
        self.shuffle_random_state = shuffle_random_state
        self.verbose = verbose
        self.debug = debug
        self.taxonomies = []  # history of taxonomies over iterations
        self.taxonomy = None  # the latest taxonomy
        self.label_set = None  # the label set used for classification

        self.attempt_0 = 0
        self.attempt_1 = 0
        self.attempt_2 = 0

        #############################
        ## Prompt templates
        #############################
        ## Phase 1: Taxonomy Generation
        #############################
        ## 1.1 Summarization
        #############################

        self.summarise_user_prompt_template = f"""
        # Instruction
        ## Context
        - **Goal**: Summarize the input text for the given use case.
        - **Data**: {self.data_description}
        - **Use case**: {self.use_case}

        # Data
        {{doc}}

        # Questions
        ## Q1. Explain how you write the summary **in {self.summary_explanation_length} words or less for this particular document and use case.**.
        ## Q2. Provide a summary of the input text **in {self.summary_length} words or less** for the use case.
        Tips:
        - The summary will represent the input data for clustering in the next step.
        - Be concise and clear. Do not add phrases like "This is the summary of the data ..." or "Summarized text: ...".
        - Within {self.summary_length} words, include the relevant information for the use case in the summary as much as possible.
        - Do not include any line breaks in the summary.
        - Provide your answer in **English** only.
        """

        #############################
        ## 1.2 Taxonomy Generation
        #############################

        ## Individual pieces of the prompts

        # Used in all three taxonomy updates to describe the requirements for the taxonomy
        self.requirements_prompt_template = f"""
        ### Format
            - Output clusters as a list of labels, where each label describes one cluster. Each label should consist of the following attributes:
                - **id**: label index starting from 1 in an incremental manner.
                - **description**: label description should be **within {cluster_description_length} words**.
                - **name**: label name should be **within {cluster_name_length} words**. It can be either *verb phrase* or *noun phrase*, whichever is more appropriate.
            - Total number of labels should be **no more than {max_num_clusters}**.
            - Output list of labels should be in **English** only.
        ### Quality
            - **No overlap or contradiction** among the categories.
            - **Name** is a concise and clear label for the cluster. Use only phrases that are specific to each cluster and avoid those that are common to all clusters. 
            - **Description** differentiates one label from another. 
            - **Name** and **description** can **accurately** and **consistently** classify new data points **without ambiguity**. 
            - **Name** and **description** are *consistent with each other*. 
            - Output clusters match the data as closely as possible, without missing important clusters or adding unnecessary ones. 
            - Output clusters serve the given use case well. 
            - Output clusters should be specific and meaningful. Do not invent labels that are not in the data.
        """

        # Used in taxonomy initial generation and refinement to describe the input batch of data
        self.summaries_data_description = """
        The input data will be a table with summaries for a list of documents, including the following columns:
            - **id**: document index.
            - **text**: document summary.
        """

        # Used in taxonomy refinement and review to describe the reference table
        self.taxonomy_data_description = """
        The input cluster table is a table with each row as a label, with the following columns:
            - **id**: label index.
            - **description**: label description used to classify data points.
            - **name**: label name.
        """

        ## Rating criteria for Q1 for updating and reviewing the taxonomy (though they have separate Q1s)
        self.rating_quality_criteria = """
        - **Intrinsic quality**:
            - 1) If the cluster table meets the *Requirements* section, with clear and consistent label names and descriptions, and no overlap or contradiction among the categories.
            - 2) If the labels in the cluster table are relevant to the given use case.
            - 3) If the cluster table includes any vague labels such as "Other", "General", "Unclear", "Miscellaneous" or "Undefined".
        - **Extrinsic quality**:
            - 1) If the cluster table can accurately and consistently classify the input data without ambiguity.
            - 2) If there are missing categories/labels in the cluster table that appear in the input data.
            - 3) If there are unnecessary categories/labels in the cluster table that do not appear in the input data.
        """

        ## Q2-4 for updating and reviewing the taxonomy
        self.q2_explain_rating = """
        ## Q2 - Explanation: Explain your rating score in Q1 **within {self.taxonomy_explanation_length} words**.
        """
        self.q3_suggest_edits = """
        ## Q3 - Edits: Based on your review, decide if you need to edit the reference table to improve its quality. If yes, suggest potential edits as precisely as possible **within {self.taxonomy_update_suggestion_length} words**. If no, please output "N/A".
        ### Tips:
        - You can edit the label name, description, or remove a label/category.
        - You can also merge or add new labels if needed. Your edits should meet the *Requirements* section.
        - The cluster table should be a **flat list** of **mutually exclusive** categories/labels. Sort them based on their semantic relatedness.
        - You can have *fewer than {self.max_num_clusters} labels* in the cluster table, but **do not exceed the limit**.
        - Be **specific** about each label/category. **Do not include vague labels** such as "Other", "General", "Unclear", "Miscellaneous" or "Undefined" in the cluster table.
        - You can ignore low-quality or ambiguous data points.
        """
        self.q4_provide_updated_labels = """
        ## Q4 - Labels: Based on your analysis, refine the taxonomy (adding, deleting, merging, splitting, sorting, editing... labels as you think necessary) and provide your updated reference table. If you have suggested to make no edits to the reference table, please output the original reference table.
        """

        ## System and user message pairs for the different stages of the pipeline
        #############################
        ## 1.2.1 Initial taxonomy generation
        #############################

        self.initial_taxonomy_system_message = "You are an expert at creating taxonomies from data for a given use-case. You create an initial label taxonomies from subsample of texts from a corpus."
        self.initial_taxonomy_user_message = f"""
        # Instruction
        ## Context
        - **Goal**: Your goal is to cluster the input data into meaningful categories for the given use case.
        - **Data**: {self.summaries_data_description}
        - **Use case**: {self.use_case}
        
        ## Requirements
        {self.requirements_prompt_template}

        # Data
        {{summaries_batch}} 
        
        # Questions 
        ## Q1. Explanation: Why do you cluster the data the way you do? Explain your reasoning ** within {self.taxonomy_explanation_length} words**. 
        ## Q2. Rating score: As this is the initial taxonomy, rate it with 0 for now. You will refine it in the next steps.
        ## Q3. Edits: Since this is the initial taxonomy, you do not need to suggest edits. Please output "N/A".
        ## Q4. Labels: Please generate a list of labels from the input data that meets the requirements.
        Tips
        - The cluster labels should be a **flat list** of **mutually exclusive** categories. Sort them based on their semantic relatedness.
        - You can have *fewer than {self.max_num_clusters} clusters* in the label list, but **do not exceed the limit.**
        - Be **specific** about each label. **Do not include vague labels** such as "Other", "General", "Unclear", "Miscellaneous" or "Undefined" in the cluster table.
        - You can ignore low quality or ambiguous data points.
        # Output
        """

        #############################
        ## 1.2.2 Iterative taxonomy refinement
        #############################

        self.update_taxonomy_system_message = (
            "You review and refine existing taxonomies based on new unseen data points."
        )
        self.update_taxonomy_user_message = f"""
        # Instruction
        ## Context
        - **Goal**: Your goal is to review the given reference table based on the input data for the specified use case, then update the reference table if needed.
            - You will be given a reference cluster table, which is built on existing data. The reference table will be used to classify new data points.
            - You will compare the input data with the reference table, output a rating score of the quality of the reference table, suggest potential edits, and update the reference table if needed.
        - **Reference cluster table**: {self.taxonomy_data_description}
        - **Data**: {self.summaries_data_description}
        - **Use case**: {self.use_case}

        ## Requirements
        {self.requirements_prompt_template}

        # Reference cluster table

        {{self.taxonomy.labels}}

        # Data

        {{summaries_batch}}

        # Questions

        ## Q1 - Rating: Review the given reference table and the input data and provide a rating score of the reference table.
        The rating score should be an integer between 0 and 100, where a higher rating score means better quality. Consider the following factors when rating the reference cluster table:
        {self.rating_quality_criteria}
        {self.q2_explain_rating}
        {self.q3_suggest_edits}
        {self.q4_provide_updated_labels}
        # Output
        """

        self.synthesise_taxonomy_system_message = (
            "You combine two proposed taxonomies into a single taxonomy which is a perfect synthesis of the two."
        )
        self.synthesise_taxonomy_user_message = f"""
        # Instruction
        ## Context
        - **Goal**: Your goal is to review the given reference tables/taxonomies based on the specified use case.
            - You will be given two reference cluster tables, which were built by independent judges by looking at distinct subsamples of the whole dataset. The reference table will be used to classify new data points.
            - You will compare the two reference tables, output an averaged rating score for the two reference tables, suggest how they can be best possibly merged, and, finally, provide the newly synthesised reference table/taxonomy.
        - **Reference cluster table**: {self.taxonomy_data_description}
        - **Data**: {self.summaries_data_description}
        - **Use case**: {self.use_case}

        ## Requirements
        {self.requirements_prompt_template}

        # Reference cluster table 1
        
        {{taxonomy1}}
        
        # Reference cluster table 2
        
        {{taxonomy2}}

        # Questions

        ## Q1 - Rating: Review the given reference tables and provide a rating score of the reference tables. Provide an averaged rating score for the two reference tables.
        The rating score should be an integer between 0 and 100, where a higher rating score means better quality. Consider the following factors when rating the each reference cluster table:
        {self.rating_quality_criteria}
        {self.q2_explain_rating}
        ## Q3 - Edits: Based on your review of the two reference tables, suggest how they can be best possibly merged to create a new reference table which better captures the whole of the data. Provide your suggestions as precisely as possible **within {self.taxonomy_update_suggestion_length} words**.
        ## Q4 - Labels: Based on your analysis, provide your synthesised taxonomy.
        # Output
        """

        #############################
        ## 1.2.3 Final taxonomy review
        #############################

        self.review_taxonomy_system_message = (
            "You review and refine existing label taxonomies based on predefined requirements."
        )
        self.review_taxonomy_user_message = f"""
        # Instruction
        ## Context
        - **Goal**: Your goal is to review the given reference table based on the requirements and the specified use case, then update the reference table if needed.
            - You will be given a reference cluster table, which is built on existing data. The reference table will be used to classify new data points.
            - You will compare the reference table with the given requirements, output a rating score of the quality of the reference table, suggest potential edits, and update the reference table if needed.
        - **Reference cluster table**: {self.taxonomy_data_description}
        - **Use case**: {self.use_case}
        ## Requirements
        {self.requirements_prompt_template}

        # Reference cluster table

        {{self.taxonomy.labels}}

        # Questions
        ## Q1 - Rating: Review the given reference table and provide a rating score. The rating score should be an integer between 0 and 100, higher rating score means better quality. You should consider the following factors when rating the reference cluster table:
        {self.rating_quality_criteria}
        {self.q2_explain_rating}
        ## Q3 - Edits: Based on your review, decide if you need to edit the reference table to improve its quality. If yes, suggest potential edits as precisely as possible **within {self.taxonomy_update_suggestion_length} words**. If no, please output "N/A".
        ### Tips:
        - You can edit the label name, description, or remove a label/category.
        - You can also merge or add new labels if needed. Your edits should meet the *Requirements* section.
        - The cluster table should be a **flat list** of **mutually exclusive** categories/labels. Sort them based on their semantic relatedness.
        - You can have *fewer than {self.max_num_clusters} labels* in the cluster table, but **do not exceed the limit**.
        - Be **specific** about each label/category. **Do not include vague labels** such as "Other", "General", "Unclear", "Miscellaneous" or "Undefined" in the cluster table.
        - You can ignore low-quality or ambiguous data points.
        ## Q4 - Labels: With full autonomy, suggest the final taxonomy based on your review and the edits you have suggested. If you have suggested to make no edits to the reference table, please output the original reference table.
        # Output
        """

        #############################
        ## Phase 2: Text Classification
        #############################

        self.label_assignment_system_message = (
            "You are an expert at classifying texts into predefined categories."
        )
        self.label_assignment_user_message = f"""
        # Instruction 
        ## Context
        - **Goal**: Your goal is to classify the input data using the provided reference table.
        - **Reference table**: The input reference table is a markdown table with each row as a category/label, with the following columns:
            - **id**: label index. 
            - **description**: label description used to classify data points.
            - **name**: label name. 
        - **Data**: {self.data_description}
        
        # Reference table 
        {{self.label_set}} 
        
        # Data
        {{doc}} 
        
        # Questions
        ## Please classify the input data using the reference table. Your output should include the following information:
        - **explanation**: a short explanation of why you think the input data belongs to the label/category or you cannot classify the data into any of the given categories/labels. Your explanation should be within {self.classification_explanation_length} words.
        - **id**: **id** of a label in the reference table; if unable to classify using the reference table , please output "-1". 
        - **name**: **name** of a label in the reference table that corresponds to the **label-id**; if unable to classify using the reference table , please output "Undefined".
        Tips
        - You should only output the **primary** label/category for the input data. If it can be classified into multiple labels/categories, please output **the most specific label/category** .
        - Your output should be in *English* only.
        """

    def get_params(self):
        return {
            "use_case": self.use_case,
            "data_description": self.data_description,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "retries": self.retries,
            "wait_time": self.wait_time,
            "summaries_max_thread_workers": self.summaries_max_thread_workers,
            "cluster_name_length": self.cluster_name_length,
            "cluster_description_length": self.cluster_description_length,
            "max_num_clusters": self.max_num_clusters,
            "summary_length": self.summary_length,
            "summary_explanation_length": self.summary_explanation_length,
            "taxonomy_explanation_length": self.taxonomy_explanation_length,
            "classification_explanation_length": self.classification_explanation_length,
            "taxonomy_update_suggestion_length": self.taxonomy_update_suggestion_length,
            "taxonomy_refinement_iterations": self.taxonomy_refinement_iterations,
            "summary_model": self.summary_model,
            "taxonomy_generation_model": self.taxonomy_generation_model,
            "synthesis_mode": self.synthesis_mode,
            "llm_classification_model": self.llm_classification_model,
            "encoder_model_name": self.encoder_model_name,
            "taxonomy_sample_ratio": self.taxonomy_sample_ratio,
            "classifier_sample_ratio": self.classifier_sample_ratio,
            "classifier_train_test_split_ratio": self.classifier_train_test_split_ratio,
            "shuffle_training_data": self.shuffle_training_data,
            "shuffle_taxonomy_data": self.shuffle_taxonomy_data,
            "shuffle_classifer_data": self.shuffle_classifer_data,
            "shuffle_random_state": self.shuffle_random_state,
            "verbose": self.verbose,
            "debug": self.debug,
        }
    
    ## Metrics calculation
    def calculate_external_cluster_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """
        Calculate topic alignment between df1 and df2 (harmonic purity, ARI, NMI...)

        Args:
            y_true: List of true labels
            y_pred: List of predicted labels

        Returns:
            Dictionary of clustering metrics
        """

        purity, inverse_purity, harmonic_purity = calculate_purity(y_pred, y_true)

        metrics = {
            "P": purity,
            "Inv_P": inverse_purity,
            "HM_P": harmonic_purity,
            "ARI": adjusted_rand_score(y_true, y_pred),
            "NMI": normalized_mutual_info_score(y_true, y_pred),
            "homogeneity": homogeneity_score(y_true, y_pred),
            "completeness": completeness_score(y_true, y_pred),
            "v_measure": v_measure_score(y_true, y_pred),
            "adjusted_mutual_info": adjusted_mutual_info_score(y_true, y_pred),
        }

        return metrics


    def calculate_internal_cluster_metrics(self, test_docs, y_pred, embeddings=None) -> Dict[str, float]:
        """
        Embeds text data, calculates internal cluster metrics, and prints the results.

        Args:
            train_data (pd.DataFrame): DataFrame with "text" and "category" columns.
        """

        if len(np.unique(y_pred)) < 2:
            print("WARNING: Only one cluster found. Skipping internal cluster metrics calculation.")
            return {"silhouette": 0, "davies_bouldin": 0, "calinski_harabasz": 0}

        # 1. Embeddings with modernbert
        if embeddings is None:
            model = SentenceTransformer(self.encoder_model_name)
            embeddings = model.encode(test_docs)

        # 2. Encode y_pred for metric calculation
        le = LabelEncoder()
        y_pred = le.fit_transform(y_pred)

        return {
            "silhouette": float(silhouette_score(embeddings, y_pred, metric="cosine")),
            "davies_bouldin": davies_bouldin_score(embeddings, y_pred),
            "calinski_harabasz": calinski_harabasz_score(embeddings, y_pred),
        }


    # ------------------------------------------------------------------------------
    # Phase 1: Taxonomy Generation
    # ------------------------------------------------------------------------------

    # ========================
    # Phase 1.1: Summarization
    # ========================

    def summarise_doc(self, doc: str) -> tuple[Any, Any]:
        summarise_messages = [
            {
                "role": "system",
                "content": "You are an expert at summarising texts for downstream analysis for a given use-case.",
            },
            {
                "role": "user",
                "content": self.summarise_user_prompt_template.format(self=self, doc=doc),
            },
        ]
        # if self.debug:
        #     print("[DEBUG] Textonomy@summarise_doc - printing variable 'summarise_messages':")
        #     print(summarise_messages)

        doc_summary = call_llm(
            self=self,
            messages=summarise_messages,
            model=self.summary_model,
            response_format=SummaryExtraction,
            temperature=self.temperature,
            max_tokens=self.max_completion_tokens,
            retries=self.retries,
            wait_time=self.wait_time,
        )
        return doc_summary.summary, doc_summary.explanation

    def generate_summaries(
        self,
        corpus: List[str],
    ) -> tuple[List[str], List[str]]:
        """
        Summarize each document in the corpus using an LLM.
        If no use_case is provided, the prompt fallback is "Please summarize for topic detection."
        """
        # Create a partial function with self bound
        summarise_doc_bound = partial(self.summarise_doc)

        summaries = []
        explanations = []

        with ThreadPoolExecutor(max_workers=self.summaries_max_thread_workers) as executor:
            for result in tqdm(
                executor.map(summarise_doc_bound, corpus),
                total=len(corpus),
                desc="Summarising documents",
            ):
                summaries.append(result[0])
                explanations.append(result[1])

        return summaries, explanations

    # ==============================
    # Phase 1.2: Taxonomy Generation
    # ==============================

    def update_taxonomy(
        self,
        summaries_batch: List[str] | None,
        system_message: str,
        user_message: str,
        taxonomy1: Optional[str] = None,
        taxonomy2: Optional[str] = None,
    ) -> Taxonomy:
        """
        Create the initial label taxonomy from the first batch of summaries.
        """
        if summaries_batch:
            messages = [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message.format(self=self, summaries_batch=summaries_batch),
                },
            ]
        elif taxonomy1 and taxonomy2:
            messages = [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message.format(self=self, taxonomy1=taxonomy1, taxonomy2=taxonomy2),
                },
            ]
        elif summaries_batch is None:
            messages = [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message.format(self=self),
                },
            ]
        elif summaries_batch == []: # no taxonomy generation data given
            messages = [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message.format(self=self, summaries_batch=summaries_batch),
                },
            ]
        else:
            print("ERROR: Textonomy@update_taxonomy: messages were not formatted.")

        if self.debug:
            print("[DEBUG] Textonomy@update_taxonomy - printing variable 'messages':")
            print(messages)

        taxonomy = call_llm(
            self=self,
            messages=messages,
            response_format=Taxonomy,
            model=self.taxonomy_generation_model,
            temperature=self.temperature,
            max_tokens=self.max_completion_tokens,
            retries=self.retries,
            wait_time=30,
        )
        return taxonomy

    # ========================
    # Phase 1 Pipeline
    # ========================

    def taxonomy_generation_pipeline(
        self,
        raw_corpus_sample: List[str],
    ) -> tuple[Taxonomy, list[Taxonomy], list[str], list[str]]:
        """
        Wrapper function that:
        1) Batches the summaries,
        2) Creates an initial taxonomy from the first batch,
        3) Iteratively updates it with subsequent batches,
        4) Optionally repeats for multiple passes (num_iterations),
        5) Reviews and finalizes the taxonomy.
        """
        if self.verbose:
            print("Phase 1 - Stage 1: Summarization")

        # LLM summarise subsample considering the use case
        self.summaries, self.explanations = self.generate_summaries(corpus=raw_corpus_sample)

        # Split summaries into batches as evenly as possible
        self.chunked = [list(batch) for batch in np.array_split(self.summaries, self.num_batches)]

        if not self.chunked:
            raise ValueError("No summaries to process.")

        self.batch_size = len(self.summaries) / self.num_batches

        if self.verbose:
            print("Summaries generated. In total ", len(self.summaries), " summaries.")
            print(
                "Split into ",
                len(self.chunked),
                " batches of ~",
                self.batch_size,
                " summaries each.",
            )
            print("Example summaries: ", self.summaries[:2])
            print("Example explanations: ", self.explanations[:2])

            print(
                "Phase 1 - Stage 2: Taxonomy Generation over ",
                self.taxonomy_refinement_iterations,
                " iterations, where the final one is a review without new data.",
            )

            print("Phase 1 - Stage 2.1: Initial creation of taxonomy")

        self.taxonomy = self.update_taxonomy(
            self.chunked[0],
            system_message=self.initial_taxonomy_system_message,
            user_message=self.initial_taxonomy_user_message,
        )
        self.taxonomies.append(self.taxonomy)

        if self.verbose:
            print("1. Initial Taxonomy:")
            self.taxonomy.print_with_dataframe()

        # Iteratively update the taxonomy
        if self.taxonomy_refinement_iterations > 2:
            if self.verbose:
                print("Phase 1 - Stage 2.2: Refining taxonomy iteratively")
            for iteration, batch in enumerate(
                self.chunked[1:]
            ):  # starting from 2nd batch because of initial creation batch that was already used
                if self.synthesis_mode:
                    self.taxonomy = self.update_taxonomy(
                        summaries_batch=batch,
                        system_message=self.initial_taxonomy_system_message,
                        user_message=self.initial_taxonomy_user_message,
                    )
                    self.taxonomies.append(self.taxonomy)
                    self.taxonomy = self.update_taxonomy(
                        summaries_batch=None,
                        system_message=self.synthesise_taxonomy_system_message,
                        user_message=self.synthesise_taxonomy_user_message,
                        taxonomy1=self.taxonomies[-1].to_pretty_json(),
                        taxonomy2=self.taxonomies[-2].to_pretty_json(),
                    )
                else:
                    self.taxonomy = self.update_taxonomy(
                        summaries_batch=batch,
                        system_message=self.update_taxonomy_system_message,
                        user_message=self.update_taxonomy_user_message,
                    )
                self.taxonomies.append(self.taxonomy)
                if self.verbose:
                    print(iteration + 2, ". Iteration on Taxonomy:")
                    self.taxonomy.print_with_dataframe()

        # Final review
        if self.taxonomy_refinement_iterations >= 2:
            print("Phase 1 - Stage 2.3: Final review of taxonomy")
            self.taxonomy = self.update_taxonomy(
                summaries_batch=None,
                system_message=self.review_taxonomy_system_message,
                user_message=self.review_taxonomy_user_message,
            )
            self.taxonomies.append(self.taxonomy)

            if self.verbose:
                print(self.taxonomy_refinement_iterations, ". Final Taxonomy:")
                self.taxonomy.print_with_dataframe()

        # Update label_set for classification
        self.label_set = [(label.id, label.name) for label in self.taxonomy.labels]

        return self.taxonomy, self.taxonomies, self.summaries, self.explanations

    # ========================
    # Phase 2 Pipeline
    # ========================

    def classifier_training_pipeline(
        self,
        train_data: List[str],
        test_data: List[str],
        label_set: List[str] = [],
    ) -> Any:
        """
        Pipeline to train a text classifier using pseudo-labels from LLM.

        Args:
            train_data: List of training text documents
            test_data: List of test text documents
            label_list: By default, the taxonomy labels are used for classification. But you can provide a custom label set to override the taxonomy labels for this pipeline.

        Returns:
            Trained classifier model
            Pseudo-labels for training data
            Explanations for training data pseudo-labels
            Pseudo-labels for test data
            Explanations for test data pseudo-labels
        """
        if label_set:
            self.label_set = [(i + 1, label) for i, label in enumerate(label_set)]
            if self.verbose:
                print(f"Using custom label set for classification: \n {self.label_set}")
        elif self.label_set is None:
            print(
                "[Error]: No label set provided and no taxonomy labels available. Please generate a taxonomy first."
            )
            return None
        else:
            if self.verbose:
                print(f"Using taxonomy labels for classification: \n {self.label_set}")

        if self.verbose:
            print("Phase 2 - Stage 1: Generating pseudo-labels using LLM")

        # Get pseudo-labels for training data
        train_label_ids, train_labels, train_explanations = self.pseudo_label_via_llm(
            texts=train_data
        )

        if self.debug:
            print(
                "[DEBUG] Textonomy@classifier_training_pipeline - printing variable 'train_labels' alongside train data:"
            )
            for i, (text, label_id, label, explanation) in enumerate(
                zip(train_data, train_label_ids, train_labels, train_explanations)
            ):
                if i >= 5:  # Limit to the first 5 for brevity
                    break
                print("Pseudo-label examples for training data:")
                print(f"--- Example {i+1} ---")
                print(f"Text: {text[:200]}...")  # Show first 100 characters
                print(f"Label ID: {label_id}")
                print(f"Label: {label}")
                print(f"Explanation: {explanation[:300]}...")  # Show first 150 characters
                print("-" * 20)

        # Get pseudo-labels for test data
        test_label_ids, test_labels, test_explanations = self.pseudo_label_via_llm(texts=test_data)

        if self.debug:
            print(
                "[DEBUG] Textonomy@classifier_training_pipeline - printing variable 'test_labels' alongside test data:"
            )
            for i, (text, label_id, label, explanation) in enumerate(
                zip(test_data, test_label_ids, test_labels, test_explanations)
            ):
                if i >= 5:  # Limit to the first 5 for brevity
                    break
                print("Pseudo-label examples for test data:")
                print(f"--- Example {i+1} ---")
                print(f"Text: {text[:200]}...")  # Show first 100 characters
                print(f"Label ID: {label_id}")
                print(f"Label: {label}")
                print(f"Explanation: {explanation[:300]}...")  # Show first 150 characters
                print("-" * 20)

        if self.verbose:
            print(
                f"Generated labels for {len(train_labels)} training and {len(test_labels)} test examples"
            )
            print("Phase 2 - Stage 2: Training classifier on pseudo-labeled data")

        # Train classifier using pseudo-labeled data
        classifier = self.train_classifier(
            train_texts=train_data,
            train_labels=train_labels,
            test_texts=test_data,
            test_labels=test_labels,
        )

        return classifier, train_labels, train_explanations, test_labels, test_explanations

    def pseudo_label_via_llm(self, texts: List[str]) -> tuple[List[int], List[int], List[str]]: #TODO: inconsistent?
        """
        Use LLM to generate pseudo-labels for texts using the taxonomy.

        Args:
            texts: List of text documents to label

        Returns:
            Tuple of (label_ids, explanations)
        """

        label_ids = []
        labels = []
        explanations = []

        # Create a partial function with self bound
        label_doc = partial(self._label_single_doc)

        if self.debug:
            print("DEBUG: Entering pseudo_label_via_llm with texts:")
            for i, text in enumerate(texts):
                if i >= 2:  # Limit to the first 2 for brevity
                    break
                print(f"--- Example {i+1} ---")
                print(f"Text: {text[:200]}...")  # Show first 200 characters
                print("-" * 20)
            print("DEBUG: End of texts.")

        with ThreadPoolExecutor(max_workers=self.summaries_max_thread_workers) as executor:
            for result in tqdm(
                executor.map(label_doc, texts), total=len(texts), desc="Generating pseudo-labels"
            ):
                label_ids.append(result[0])
                labels.append(result[1])
                explanations.append(result[2])

        return label_ids, labels, explanations

    def _label_single_doc(self, doc: str) -> tuple[int, str, str]:
        """Helper method to label a single document using LLM."""

        messages = [
            {"role": "system", "content": self.label_assignment_system_message},
            {
                "role": "user",
                "content": self.label_assignment_user_message.format(self=self, doc=doc),
            },
        ]

        response = call_llm(
            self=self,
            messages=messages,
            model=self.llm_classification_model,
            response_format=DocClassification,
            temperature=self.temperature,
            max_tokens=self.max_completion_tokens,
            retries=self.retries,
            wait_time=self.wait_time,
        )

        return response.id, response.name, response.explanation

    def train_classifier(
        self,
        train_texts: List[str],
        train_labels,
        test_texts: List[str],
        test_labels,
        encoder_model_name=None,
        random_state: int = 42,
        max_iter: int = 1000,
        classifier_type: Literal["LogisticRegression", "XGBoost"] = "LogisticRegression",
        xgb_params: Optional[Dict[str, Any]] = None,
        handle_imbalance: bool = True,  # Add a parameter to control imbalance handling
        train_embeddings: Optional[np.ndarray] = None,
        test_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Trains a text classifier, handling class imbalance.

        Args:
            train_texts: Training text documents.
            train_labels: Training labels (string).
            test_texts: Test text documents.
            test_labels: Test labels (string).
            encoder_model_name: Sentence Transformer model name.
            random_state: Random seed.
            max_iter: Max iterations for LogisticRegression.
            classifier_type: 'LogisticRegression' or 'XGBoost'.
            xgb_params: XGBoost parameters.
            handle_imbalance: Whether to handle class imbalance.

        Returns:
            Dictionary with model, encoder, metrics, classes, and label encoder.
        """
        if encoder_model_name:
            self.encoder_model_name = encoder_model_name
        if self.verbose:
            print(f"Loading Sentence Transformer model: {self.encoder_model_name}")

        encoder = SentenceTransformer(self.encoder_model_name)

        # Collect allowed labels from taxonomy (add "Undefined" for fallback, if not present)
        # Extract label names from tuples (id, name)
        allowed_label_names = [name for _, name in self.label_set]
        if "Undefined" not in allowed_label_names:
            allowed_label_names.append("Undefined")
        
        if self.debug:
            print(f"[DEBUG] Allowed label names for filtering: {allowed_label_names}")

        def _filter_invalid_samples(texts, labels, split_name, embeddings=None):
            keep_indices = [i for i, label in enumerate(labels) if label in allowed_label_names]
            dropped = len(labels) - len(keep_indices)

            if dropped:
                print(
                    f"[Warning] Dropped {dropped} {split_name} sample(s) with labels not in the taxonomy."
                )

            if not keep_indices:
                raise ValueError(
                    f"No {split_name} samples remain after removing labels outside the taxonomy."
                )

            filtered_texts = [texts[i] for i in keep_indices]
            filtered_labels = [labels[i] for i in keep_indices]

            if embeddings is None:
                filtered_embeddings = None
            else:
                filtered_embeddings = np.asarray(embeddings)[keep_indices]

            return filtered_texts, filtered_labels, filtered_embeddings

        train_texts, train_labels, train_embeddings = _filter_invalid_samples(
            texts=train_texts,
            labels=list(train_labels),
            split_name="training",
            embeddings=train_embeddings,
        )

        test_texts, test_labels, test_embeddings = _filter_invalid_samples(
            texts=test_texts,
            labels=list(test_labels),
            split_name="test",
            embeddings=test_embeddings,
        )

        label_encoder = LabelEncoder()
        label_encoder.fit(allowed_label_names)  # Fit only on taxonomy label names + Undefined
        train_labels_encoded = label_encoder.transform(train_labels)
        test_labels_encoded = label_encoder.transform(test_labels)
        self.label_encoder = label_encoder

        if self.verbose:
            print("Generating embeddings for training data...")
        if train_embeddings is None:
            train_embeddings = encoder.encode(train_texts, show_progress_bar=self.verbose)

        if self.verbose:
            print("Generating embeddings for test data...")
        if test_embeddings is None:
            test_embeddings = encoder.encode(test_texts, show_progress_bar=self.verbose)

        if classifier_type:
            self.classifier_model = classifier_type
        if self.verbose:
            print(f"Training {self.classifier_model} classifier...")

        if handle_imbalance:
            unique_classes = np.unique(train_labels_encoded)
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=unique_classes,
                y=train_labels_encoded,
            )
            # Map each unique class to its corresponding weight
            class_weights_dict = dict(zip(unique_classes, class_weights))
        else:
            class_weights_dict = None  # No class weighting

        if self.classifier_model == "LogisticRegression":
            self.classifier = LogisticRegression(
                random_state=random_state,
                max_iter=max_iter,
                n_jobs=-1,
                class_weight=class_weights_dict,  # Use class_weight parameter
            )
            self.classifier.fit(train_embeddings, train_labels_encoded)
            self.classifier_signature = infer_signature(train_embeddings, train_labels_encoded)
            self.classifier_input_example = train_embeddings[:3]

        elif self.classifier_model == "XGBoost":
            train_embeddings = np.array(train_embeddings)
            train_labels_encoded = np.array(train_labels_encoded)
            test_embeddings = np.array(test_embeddings)
            test_labels_encoded = np.array(test_labels_encoded)

            default_xgb_params = {
                "objective": "binary:logistic"
                if len(set(train_labels_encoded)) == 2
                else "multi:softmax",
                "eval_metric": "logloss"
                if len(set(train_labels_encoded)) == 2
                else "mlogloss",
                "num_class": len(set(train_labels_encoded))
                if len(set(train_labels_encoded)) > 2
                else None,
                "seed": random_state,
                "n_jobs": -1,
                "use_label_encoder": False,
            }

            if xgb_params:
                default_xgb_params.update(xgb_params)
            
            #Handle class_weights for XGBoost
            if handle_imbalance and class_weights_dict:
                #XGBoost expects sample weights, not class weights. Calculate sample weights
                sample_weights = np.array([class_weights_dict[label] for label in train_labels_encoded]) # type: ignore
                default_xgb_params["sample_weight"] = sample_weights # Pass sample_weight to fit()
                
            self.classifier = xgb.XGBClassifier(**default_xgb_params)


            if handle_imbalance and class_weights_dict:  # Pass sample_weights to fit() if handling imbalance
                self.classifier.fit(train_embeddings, train_labels_encoded, sample_weight=sample_weights)
            else:
                self.classifier.fit(train_embeddings, train_labels_encoded)
                
            self.classifier_signature = infer_signature(train_embeddings, train_labels_encoded)
            self.classifier_input_example = train_embeddings[:3]

        else:
            raise ValueError(
                f"Invalid classifier_type: {self.classifier_model}.  Must be 'LogisticRegression' or 'XGBoost'."
            )

        test_predictions = self.classifier.predict(test_embeddings)

        report = classification_report(
            y_true=test_labels_encoded, y_pred=test_predictions, output_dict=True
        )

        if self.verbose:
            print("\nTest Set Performance:")
            print(classification_report(y_true=test_labels_encoded, y_pred=test_predictions))

        self.encoder = encoder
        self.classes_ = (
            self.classifier.classes_
            if hasattr(self.classifier, "classes_")
            else np.unique(train_labels_encoded)
        )
        if len(self.classes_) < len(self.label_set):
            print(
                f"[Warning] Only {len(self.classes_)} classes found in training data out of {len(self.label_set)} taxonomy labels (+ 'Undefined')."
            )
        self.metrics_ = report
        self.accuracy_ = accuracy_score(y_true=test_labels_encoded, y_pred=test_predictions)
        self.f1_score_ = f1_score(
            y_true=test_labels_encoded, y_pred=test_predictions, average="weighted"
        )

        self.confusion_matrix_ = confusion_matrix(
            y_true=test_labels_encoded, y_pred=test_predictions, labels=self.classes_
        )
        self.confusion_matrix_display = ConfusionMatrixDisplay(
            confusion_matrix=self.confusion_matrix_,
            display_labels=self.label_encoder.inverse_transform(self.classes_),
        ).plot(cmap="Blues")

        return {
            "model": self.classifier,
            "encoder": encoder,
            "metrics": report,
            "classes": self.classes_,
            "label_encoder": self.label_encoder,
        }


    def fit(self, X: List[str]) -> "Textonomy":
        """
        Fit the Textonomy model by:
        1. Generating taxonomy from a sample of data
        2. Training classifier on pseudo-labeled data

        Args:
            X: List of text documents to train on

        Returns:
            self: The fitted model instance
        """
        if self.verbose:
            print("Starting Textonomy fitting process...")

        # Calculate sample sizes and split data
        taxonomy_sample_size = int(self.taxonomy_sample_ratio * len(X))
        print(f"Taxonomy sample size: {taxonomy_sample_size}")
        classifier_sample_size = int(self.classifier_sample_ratio * len(X))
        print(f"Classifier sample size: {classifier_sample_size}")

        if self.shuffle_training_data:
            if self.shuffle_random_state:
                random.Random(self.shuffle_random_state).shuffle(X)
                taxonomy_data, _ = train_test_split(
                    X,
                    train_size=taxonomy_sample_size,
                    shuffle=self.shuffle_taxonomy_data,
                    random_state=self.shuffle_random_state,
                )
                train_data, test_data = train_test_split(
                    X[:classifier_sample_size],
                    train_size=self.classifier_train_test_split_ratio,
                    shuffle=self.shuffle_classifer_data,
                )
            else:
                random.shuffle(X)
                taxonomy_data, _ = train_test_split(
                    X,
                    train_size=taxonomy_sample_size,
                    shuffle=self.shuffle_taxonomy_data,
                    random_state=self.shuffle_random_state,
                )
                train_data, test_data = train_test_split(
                    X[:classifier_sample_size],
                    train_size=self.classifier_train_test_split_ratio,
                    shuffle=self.shuffle_classifer_data,
                )

        # Split into train/test sets
        # make taxonomy data a randomly drawn sample of X

        if self.verbose:
            print(f"Generating taxonomy using {len(taxonomy_data)} documents...")

        # Generate taxonomy
        self.taxonomy_generation_pipeline(raw_corpus_sample=taxonomy_data)

        if self.verbose:
            print(
                f"Training classifier using {len(train_data)} training and {len(test_data)} test documents..."
            )

        # Train classifier
        self.classifier_training_pipeline(train_data=train_data, test_data=test_data)

        if self.verbose:
            print("Fitting complete!")
            print(f"Model performance on test set:\n{self.metrics_}")

        return self

    def predict(self, X: List[str]) -> List[str]:
        """
        Predict labels for new documents using the trained classifier.

        Args:
            X: List of text documents to classify

        Returns:
            List of predicted labels
        """
        if not hasattr(self, "classifier") or not hasattr(self, "encoder"):
            raise RuntimeError("Model hasn't been fitted. Call fit() first.")

        if self.verbose:
            print(f"Generating predictions for {len(X)} documents...")

        # Generate embeddings
        embeddings = self.encoder.encode(X, show_progress_bar=self.verbose)

        # Get predictions
        predictions = self.classifier.predict(embeddings)

        # Calculate internal cluster metrics
        self.internal_cluster_metrics = self.calculate_internal_cluster_metrics(X, predictions, embeddings)

        # Convert numeric predictions to label names
        # label_map = {i: name for i, name in self.label_set}
        # predicted_labels = [label_map[pred] for pred in predictions]

        return predictions.tolist()
