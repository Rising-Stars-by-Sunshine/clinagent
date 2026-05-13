import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

# ==========================================
# 1. 初始化异构模型实例 (Ollama)
# ==========================================

# 感知模型：使用 Llama 3.1，擅长自然语言描述与语义对齐
perception_llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

# 推理模型：使用 DeepSeek-R1，利用其强大的 CoT 思维链进行临床逻辑分析
reasoning_llm = ChatOllama(model="deepseek-r1:8b", temperature=0.2)

# 审计模型：使用 Qwen 2.5，侧重于事实校验与逻辑纠错，减少偏见
audit_llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# ==========================================
# 2. 定义智能体节点 (Nodes)
# ==========================================

def perception_node(state):
    """
    感知智能体：融合文本与 AU 数值，生成临床日志
    """
    prompt = f"结合文本: {state['transcript_text']} 和视觉特征: {state['au_summary']}，生成一份简短的临床表现日志。"
    
    # 调用专门的感知模型
    res = perception_llm.invoke([SystemMessage(content=prompt)])
    
    # 过滤掉可能存在的思维链标签
    content = res.content.split("</think>")[-1].strip()
    return {"perception_log": content}


def knowledge_node(state):
    """
    知识智能体：静态注入 PHQ-8 临床标准
    """
    knowledge = (
        "PHQ-8 标准：0-无，1-几天，2-一半以上时间，3-几乎每天。评分维度包括：\n"
        "1. 兴趣减退；2. 情绪低落；3. 睡眠障碍；4. 能量不足；\n"
        "5. 食欲改变；6. 自我评价低；7. 注意力集中困难；8. 行动迟缓或烦躁。"
    )
    return {"knowledge_base": knowledge}


def reasoning_node(state):
    """
    推理智能体：执行细粒度打分与证据溯源
    """
    prompt = f"""你是一位资深精神科医生。请根据以下资料进行 PHQ-8 评估：
    【观察日志】：{state['perception_log']}
    【临床标准】：{state['knowledge_base']}
    
    任务要求：
            1. 识别隐晦证据：不仅要找关键词，还要识别语义。例如“没力气/很累”对应“能量不足”；“睡不踏实/早醒”对应“睡眠障碍”。
            2. 综合 AU 表现：如果文本模糊但视觉特征（AU）显示面部活动极低（如 AU12 频率极低且持续时间短），应结合考虑。
            3. 证据分级：
            - 如果完全没提到：0分。
            - 如果有提及但程度轻或偶尔：1分。
            - 如果有明确且持续的表现：2-3分。

    4. 必须输出 JSON 格式，结构如下：
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
        "rationale": "综合所有证据的最终诊断逻辑"
    }}"""
    
    # 调用核心推理模型 DeepSeek-R1
    res = reasoning_llm.invoke([SystemMessage(content=prompt)])
    
    # 提取 JSON 内容，过滤思考过程
    content = res.content.split("</think>")[-1].strip()
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        return {"analysis_output": json.loads(content[start:end])}
    except:
        return {"analysis_output": {"error": "JSON解析失败", "raw": content}}


def audit_node(state):
    """
    审计智能体：执行逻辑一致性审查与反幻觉校验
    """
    prompt = f"""你是一位临床审计员。请严厉审查以下诊断：
    【诊断结论】：{json.dumps(state['analysis_output'])}
    【原始证据】：{state['perception_log']}
    
    审计要点：
    1. 逻辑断裂：模型是否将普通的“疲劳”直接关联到了“抑郁”？
    2. 证据失效：引用的【证据原文】是否真的支撑该得分？
    3. 倾向性偏差：模型是否在没有明确负面证据的情况下给出了非 0 分？

    要求：
    - 如果发现证据不足以支撑预测的高分，请回复：'REJECT: [具体原因]'。
    - 如果推理逻辑完全自洽且证据真实，回复 'PASS'。"""
    
    # 调用客观审计模型 Qwen
    res = audit_llm.invoke([SystemMessage(content=prompt)])
    
    content = res.content.split("</think>")[-1].strip()
    return {"audit_comment": content}