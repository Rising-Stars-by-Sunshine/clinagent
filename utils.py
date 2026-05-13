import pandas as pd
import os

def load_participant_data(data_root, p_id):
    """
    读取并预处理单个参与者的多模态数据 [cite: 1550, 1552]
    """
    # 1. 处理 Transcript (文本模态) 
    t_filename = f"{p_id}_TRANSCRIPT.csv"
    t_path = os.path.join(data_root, t_filename)
    
    # 检查文件是否存在，避免程序崩溃
    if not os.path.exists(t_path):
        return "", f"错误: 找不到文本文件 {t_path}"

    try:
        # DAIC-WOZ 原始 Transcript 通常是制表符 (\t) 分隔 
        df_t = pd.read_csv(t_path, sep='\t').fillna("")
        # 如果读取后发现没有内容或格式不对，尝试默认逗号分隔
        if 'speaker' not in df_t.columns:
            df_t = pd.read_csv(t_path).fillna("")
            
        # 仅提取参与者的发言内容 
        p_text = " ".join(df_t[df_t['speaker'] == 'Participant']['value'].astype(str).tolist())
    except Exception as e:
        p_text = f"解析 Transcript 出错: {str(e)}"
    
    # 2. 处理 AU 特征 (视觉模态) [cite: 1553]
    # 修复点：直接使用 data_root，不再使用未定义的 folder_path
    au_filename = f"{p_id}_CLNF_AUs.csv"
    au_path = os.path.join(data_root, au_filename)
    
    if not os.path.exists(au_path):
        au_summary = "视觉特征数据缺失"
    else:
        try:
            df_au = pd.read_csv(au_path)
            # 过滤掉跟踪失败的帧以保证数据质量 [cite: 1556]
            valid_au = df_au[df_au['success'] == 1]
            
            # 计算统计摘要，用于减少 Agent 的 Token 消耗
            # AU12 均值代表微笑强度，AU04 频率代表皱眉/压力频率 [cite: 1553]
            au12_mean = valid_au['AU12_r'].mean() if 'AU12_r' in valid_au else 0
            au04_freq = (valid_au['AU04_c'] == 1).mean() if 'AU04_c' in valid_au else 0
            
            au_summary = f"AU12 (微笑强度) 均值: {au12_mean:.2f}, AU04 (皱眉频率): {au04_freq:.2%}"
        except Exception as e:
            au_summary = f"解析 AU 数据出错: {str(e)}"
    return p_text, au_summary
    