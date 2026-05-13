import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from tqdm import tqdm

from knowledge_base import KnowledgeRetriever
from utils import load_participant_data


PHQ_ITEMS = {
    "1_interest": "Anhedonia",
    "2_mood": "Depressed mood",
    "3_sleep": "Sleep disturbance",
    "4_energy": "Fatigue or low energy",
    "5_appetite": "Appetite change",
    "6_self_esteem": "Low self-esteem or worthlessness",
    "7_concentration": "Concentration difficulty",
    "8_movement": "Psychomotor agitation or retardation",
}


ABLATION_CONFIGS = {
    "full_pipeline": {
        "use_perception": True,
        "use_knowledge": True,
        "use_retrieval": True,
        "use_audit": True,
    },
    "audit_only": {
        "use_perception": True,
        "use_knowledge": False,
        "use_retrieval": False,
        "use_audit": True,
    },
    "knowledge_no_rag": {
        "use_perception": True,
        "use_knowledge": True,
        "use_retrieval": False,
        "use_audit": False,
    },
    "basic_core_only": {
        "use_perception": True,
        "use_knowledge": False,
        "use_retrieval": False,
        "use_audit": False,
    },
}


@dataclass
class ExperimentModels:
    perception_model: str = "llama3.1:8b"
    knowledge_model: str = "llama3.1:8b"
    reasoning_model: str = "deepseek-r1:8b"
    audit_model: str = "qwen2.5:7b"
    temperature: float = 0.2


def create_llm(model: str, temperature: float, seed: int) -> ChatOllama:
    try:
        return ChatOllama(model=model, temperature=temperature, seed=seed)
    except TypeError:
        return ChatOllama(model=model, temperature=temperature)


def strip_reasoning_tags(content: str) -> str:
    return content.split("</think>")[-1].strip()


def parse_json_response(content: str) -> Dict[str, Any]:
    cleaned = strip_reasoning_tags(content)
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found in model response: {cleaned}")
    return json.loads(cleaned[start:end])


def normalize_prediction(output: Dict[str, Any]) -> Dict[str, Any]:
    individual_scores = output.get("individual_scores", {})
    normalized_scores = {}
    total = 0

    for item_key in PHQ_ITEMS:
        item_output = individual_scores.get(item_key, {})
        score = item_output.get("score", 0) if isinstance(item_output, dict) else 0
        try:
            score = int(round(float(score)))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(3, score))
        total += score
        normalized_scores[item_key] = {
            "score": score,
            "evidence": item_output.get("evidence", "") if isinstance(item_output, dict) else "",
        }

    output["individual_scores"] = normalized_scores
    output["total"] = max(0, min(24, int(output.get("total", total) or total)))
    output["rationale"] = output.get("rationale", "")
    return output


def phq_scoring_schema() -> str:
    item_schema = ",\n".join(
        f'            "{item_key}": {{"score": 0, "evidence": "supporting transcript evidence"}}'
        for item_key in PHQ_ITEMS
    )
    return f"""{{
        "individual_scores": {{
{item_schema}
        }},
        "total": 0,
        "rationale": "brief evidence-grounded diagnostic rationale"
    }}"""


def run_perception_agent(transcript_text: str, au_summary: str, llm: ChatOllama) -> str:
    prompt = (
        "Summarize clinically relevant observations from the patient transcript and visual AU summary. "
        "Focus on evidence relevant to PHQ-8 symptoms and avoid unsupported interpretation.\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        f"Visual AU summary:\n{au_summary}"
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    return strip_reasoning_tags(response.content)


def get_static_knowledge() -> str:
    return (
        "PHQ-8 scoring criteria: 0 = not at all, 1 = several days, "
        "2 = more than half the days, 3 = nearly every day. Items include "
        "anhedonia, depressed mood, sleep disturbance, fatigue or low energy, "
        "appetite change, low self-esteem or worthlessness, concentration difficulty, "
        "and psychomotor agitation or retardation."
    )


def run_reasoning_agent(
    observation: str,
    knowledge_context: str,
    llm: ChatOllama,
) -> Dict[str, Any]:
    prompt = f"""You are an experienced clinical evaluator. Predict the PHQ-8 score from the evidence below.

Observation:
{observation}

Clinical knowledge:
{knowledge_context}

Rules:
1. Score each PHQ-8 item from 0 to 3.
2. Cite transcript-grounded evidence for every non-zero score.
3. Do not infer symptoms that are not supported by the transcript or clinical knowledge.
4. Return only valid JSON using this structure:
{phq_scoring_schema()}"""
    response = llm.invoke([SystemMessage(content=prompt)])
    return normalize_prediction(parse_json_response(response.content))


def run_audit_agent(
    prediction: Dict[str, Any],
    observation: str,
    knowledge_context: str,
    llm: ChatOllama,
) -> Dict[str, Any]:
    prompt = f"""You are a clinical audit agent. Check whether the prediction is supported by the evidence.

Prediction:
{json.dumps(prediction)}

Observation:
{observation}

Clinical knowledge:
{knowledge_context}

Return only valid JSON with this structure:
{{
    "decision": "PASS",
    "corrected_prediction": null,
    "reason": "brief audit rationale",
    "unsupported_inference": false
}}

Use "REJECT" when the score contains unsupported symptom inference, fabricated evidence, or unjustified severity escalation. If rejected, provide a corrected_prediction using the same PHQ-8 JSON schema as the original prediction."""
    response = llm.invoke([SystemMessage(content=prompt)])
    audit_output = parse_json_response(response.content)

    decision = str(audit_output.get("decision", "PASS")).upper()
    corrected_prediction = audit_output.get("corrected_prediction")
    if decision == "REJECT" and isinstance(corrected_prediction, dict):
        audit_output["corrected_prediction"] = normalize_prediction(corrected_prediction)
    else:
        audit_output["corrected_prediction"] = None

    audit_output["decision"] = decision
    audit_output["unsupported_inference"] = bool(audit_output.get("unsupported_inference", decision == "REJECT"))
    audit_output["reason"] = audit_output.get("reason", "")
    return audit_output


def run_single_agent_case(
    transcript_text: str,
    au_summary: str,
    model_name: str,
    temperature: float,
    seed: int,
) -> Dict[str, Any]:
    llm = create_llm(model_name, temperature, seed)
    prompt = f"""You are an experienced clinical evaluator. Predict the PHQ-8 score from the patient data.

Transcript:
{transcript_text}

Visual AU summary:
{au_summary}

Return only valid JSON using this structure:
{phq_scoring_schema()}"""
    response = llm.invoke([SystemMessage(content=prompt)])
    prediction = normalize_prediction(parse_json_response(response.content))
    return {
        "prediction": prediction,
        "initial_prediction": prediction,
        "audit": {},
        "knowledge_references": [],
        "observation": transcript_text,
    }


def run_multi_agent_case(
    transcript_text: str,
    au_summary: str,
    config_name: str,
    models: ExperimentModels,
    seed: int,
) -> Dict[str, Any]:
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}")

    config = ABLATION_CONFIGS[config_name]
    perception_llm = create_llm(models.perception_model, models.temperature, seed)
    reasoning_llm = create_llm(models.reasoning_model, models.temperature, seed)
    audit_llm = create_llm(models.audit_model, models.temperature, seed)

    if config["use_perception"]:
        observation = run_perception_agent(transcript_text, au_summary, perception_llm)
    else:
        observation = f"Transcript: {transcript_text}\nVisual AU summary: {au_summary}"

    knowledge_references: List[dict] = []
    if config["use_knowledge"] and config["use_retrieval"]:
        retriever = KnowledgeRetriever(top_k=3)
        retrieved_context, knowledge_references = retriever.retrieve(observation)
        knowledge_context = f"{get_static_knowledge()}\nRetrieved references:\n{retrieved_context}"
    elif config["use_knowledge"]:
        knowledge_context = get_static_knowledge()
    else:
        knowledge_context = "No external clinical knowledge module is active."

    initial_prediction = run_reasoning_agent(observation, knowledge_context, reasoning_llm)
    final_prediction = initial_prediction
    audit_output: Dict[str, Any] = {}

    if config["use_audit"]:
        audit_output = run_audit_agent(initial_prediction, observation, knowledge_context, audit_llm)
        if audit_output.get("corrected_prediction"):
            final_prediction = audit_output["corrected_prediction"]

    return {
        "prediction": final_prediction,
        "initial_prediction": initial_prediction,
        "audit": audit_output,
        "knowledge_references": knowledge_references,
        "observation": observation,
    }


def resolve_columns(df: pd.DataFrame) -> tuple:
    id_column = "Participant_ID" if "Participant_ID" in df.columns else df.columns[0]
    score_column = "PHQ8_Score" if "PHQ8_Score" in df.columns else "PHQ8_Score"
    if score_column not in df.columns:
        raise ValueError(f"PHQ8_Score column not found. Available columns: {list(df.columns)}")
    return id_column, score_column


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def run_experiment(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    split_df = pd.read_csv(args.split_csv)
    id_column, score_column = resolve_columns(split_df)

    seeds = [int(seed) for seed in args.seeds.split(",")]
    models = ExperimentModels(
        perception_model=args.perception_model,
        reasoning_model=args.reasoning_model,
        audit_model=args.audit_model,
        temperature=args.temperature,
    )

    jsonl_path = os.path.join(args.output_dir, "case_results.jsonl")
    csv_path = os.path.join(args.output_dir, "case_results.csv")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    flat_records = []
    total_iterations = len(seeds) * len(split_df)
    progress = tqdm(total=total_iterations, desc=f"Running {args.mode}", unit="case")

    for seed in seeds:
        for _, row in split_df.iterrows():
            participant_id = int(row[id_column])
            true_score = int(row[score_column])
            transcript_text, au_summary = load_participant_data(args.data_root, participant_id)

            try:
                if args.mode == "single_agent":
                    result = run_single_agent_case(
                        transcript_text,
                        au_summary,
                        args.single_agent_model,
                        args.temperature,
                        seed,
                    )
                    configuration = f"single_agent_{args.single_agent_model}"
                else:
                    result = run_multi_agent_case(
                        transcript_text,
                        au_summary,
                        args.config,
                        models,
                        seed,
                    )
                    configuration = args.config

                prediction = result["prediction"]
                initial_prediction = result["initial_prediction"]
                audit = result.get("audit", {})
                knowledge_references = result.get("knowledge_references", [])

                record = {
                    "split": args.split_name,
                    "seed": seed,
                    "mode": args.mode,
                    "configuration": configuration,
                    "participant_id": participant_id,
                    "true_score": true_score,
                    "predicted_score": prediction["total"],
                    "initial_predicted_score": initial_prediction["total"],
                    "audit_decision": audit.get("decision", ""),
                    "audit_corrected": bool(audit.get("corrected_prediction")),
                    "hallucination_flag": bool(audit.get("unsupported_inference", False)),
                    "audit_reason": audit.get("reason", ""),
                    "knowledge_reference_ids": [item["id"] for item in knowledge_references],
                    "knowledge_retrieval_method": knowledge_references[0]["retrieval_method"]
                    if knowledge_references
                    else "",
                    "rationale": prediction.get("rationale", ""),
                    "prediction_json": prediction,
                }
            except Exception as error:
                record = {
                    "split": args.split_name,
                    "seed": seed,
                    "mode": args.mode,
                    "configuration": args.config if args.mode != "single_agent" else args.single_agent_model,
                    "participant_id": participant_id,
                    "true_score": true_score,
                    "predicted_score": None,
                    "initial_predicted_score": None,
                    "audit_decision": "ERROR",
                    "audit_corrected": False,
                    "hallucination_flag": False,
                    "audit_reason": str(error),
                    "knowledge_reference_ids": [],
                    "knowledge_retrieval_method": "",
                    "rationale": "",
                    "prediction_json": {},
                }

            append_jsonl(jsonl_path, record)
            flat_record = dict(record)
            flat_record["knowledge_reference_ids"] = ";".join(record["knowledge_reference_ids"])
            flat_record["prediction_json"] = json.dumps(record["prediction_json"], ensure_ascii=True)
            flat_records.append(flat_record)
            progress.update(1)

    progress.close()
    pd.DataFrame(flat_records).to_csv(csv_path, index=False)
    print(f"Saved case-level results to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PHQ-8 single-agent, multi-agent, and ablation experiments.")
    parser.add_argument("--data-root", required=True, help="Directory containing transcript and AU files.")
    parser.add_argument("--split-csv", required=True, help="CSV split file to evaluate.")
    parser.add_argument("--split-name", default="test", help="Name of the evaluated split.")
    parser.add_argument("--output-dir", default="results", help="Directory for experiment outputs.")
    parser.add_argument("--mode", choices=["single_agent", "multi_agent"], default="multi_agent")
    parser.add_argument("--config", choices=list(ABLATION_CONFIGS.keys()), default="full_pipeline")
    parser.add_argument("--seeds", default="11,22,33", help="Comma-separated inference seeds.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--single-agent-model", default="deepseek-r1:8b")
    parser.add_argument("--perception-model", default="llama3.1:8b")
    parser.add_argument("--reasoning-model", default="deepseek-r1:8b")
    parser.add_argument("--audit-model", default="qwen2.5:7b")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
