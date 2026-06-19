import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

# 1. Initialize heterogeneous model instances with Ollama.

# Perception model: handles natural-language description and semantic alignment.
perception_llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

# Reasoning model: performs clinical logic analysis.
reasoning_llm = ChatOllama(model="deepseek-r1:8b", temperature=0.2)

# Audit model: checks factual consistency and reduces unsupported conclusions.
audit_llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# 2. Define graph nodes.

def perception_node(state):
    """
    Generate a concise clinical observation log from transcript and AU features.
    """
    prompt = (
        "Generate a concise clinical observation log by integrating the "
        f"transcript: {state['transcript_text']} and visual AU features: "
        f"{state['au_summary']}."
    )
    
    # Invoke the perception model.
    res = perception_llm.invoke([SystemMessage(content=prompt)])
    
    # Remove optional reasoning tags from models that expose them.
    content = res.content.split("</think>")[-1].strip()
    return {"perception_log": content}


def knowledge_node(state):
    """
    Provide PHQ-8 clinical scoring criteria.
    """
    knowledge = (
        "PHQ-8 scoring criteria: 0 = not at all, 1 = several days, "
        "2 = more than half the days, 3 = nearly every day. Items include:\n"
        "1. Anhedonia; 2. Depressed mood; 3. Sleep disturbance; "
        "4. Fatigue or low energy;\n"
        "5. Appetite change; 6. Low self-esteem or worthlessness; "
        "7. Concentration difficulty; 8. Psychomotor agitation or retardation."
    )
    return {"knowledge_base": knowledge}


def reasoning_node(state):
    """
    Perform item-level PHQ-8 scoring with evidence attribution.
    """
    prompt = f"""You are an experienced psychiatrist. Conduct a PHQ-8 assessment using the following materials:
    Observation log: {state['perception_log']}
    Clinical criteria: {state['knowledge_base']}
    
    Requirements:
            1. Identify implicit evidence. Do not rely only on keywords; infer clinically relevant semantics. For example, "no strength" or "very tired" corresponds to low energy, and "restless sleep" or "early awakening" corresponds to sleep disturbance.
            2. Integrate AU evidence. If the transcript is ambiguous but the visual AU features indicate very low facial activity, such as very low AU12 frequency and short duration, consider this as supporting context.
            3. Grade evidence:
            - If an item is not mentioned at all, assign 0.
            - If it is mentioned mildly or occasionally, assign 1.
            - If it is clearly present and persistent, assign 2 or 3.

    4. Return only valid JSON in the following structure:
    {{
        "individual_scores": {{
            "1_interest": {{"score": 0, "evidence": "..."}},
            "2_mood": {{"score": 0, "evidence": "..."}},
            "3_sleep": {{"score": 0, "evidence": "..."}},
            "4_energy": {{"score": 0, "evidence": "..."}},
            "5_appetite": {{"score": 0, "evidence": "..."}},
            "6_self_esteem": {{"score": 0, "evidence": "..."}},
            "7_concentration": {{"score": 0, "evidence": "..."}},
            "8_movement": {{"score": 0, "evidence": "..."}}
        }},
        "total": 0,
        "rationale": "Final diagnostic reasoning based on all evidence"
    }}"""
    
    # Invoke the core reasoning model.
    res = reasoning_llm.invoke([SystemMessage(content=prompt)])
    
    # Extract JSON content after removing optional reasoning traces.
    content = res.content.split("</think>")[-1].strip()
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        return {"analysis_output": json.loads(content[start:end])}
    except:
        return {"analysis_output": {"error": "JSON parsing failed", "raw": content}}


def audit_node(state):
    """
    Review logical consistency and check for unsupported conclusions.
    """
    prompt = f"""You are a clinical auditor. Critically review the following diagnosis:
    Diagnostic conclusion: {json.dumps(state['analysis_output'])}
    Original evidence: {state['perception_log']}
    
    Audit criteria:
    1. Logical gaps: did the model directly equate ordinary fatigue with depression?
    2. Evidence validity: does the cited evidence actually support the assigned score?
    3. Bias toward positive scoring: did the model assign a non-zero score without explicit negative evidence?

    Required response:
    - If the evidence is insufficient to support a high predicted score, reply with: 'REJECT: [specific reason]'.
    - If the reasoning is internally consistent and evidence-based, reply with 'PASS'."""
    
    # Invoke the audit model.
    res = audit_llm.invoke([SystemMessage(content=prompt)])
    
    content = res.content.split("</think>")[-1].strip()
    return {"audit_comment": content}