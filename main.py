import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from utils import load_participant_data
from graph import app

# 路径配置
DATA_ROOT = r"D:\D\DKU\Dachuang\Dachuang24fall\Dataset" 
DEV_CSV = os.path.join(DATA_ROOT, "data_split_Depression_AVEC2017.csv")
REPORT_FILE = "clinical_diagnosis_report2.txt"

# PHQ-8 维度显示名称映射表
DISSECT_MAP = {
    "1_interest": "兴趣减退 (Anhedonia)",
    "2_mood": "情绪低落 (Depressed Mood)",
    "3_sleep": "睡眠障碍 (Sleep Disturbance)",
    "4_energy": "疲劳感 (Fatigue/Low Energy)",
    "5_appetite": "食欲改变 (Appetite Change)",
    "6_self_esteem": "自我评价低 (Guilt/Worthlessness)",
    "7_concentration": "注意力集中困难 (Concentration Issues)",
    "8_movement": "行动迟缓/烦躁 (Psychomotor Agitation/Retardation)"
}

def main():
    if not os.path.exists(DEV_CSV):
        print(f"❌ 找不到索引文件: {DEV_CSV}")
        return

    dev_df = pd.read_csv(DEV_CSV)
    test_df = dev_df.head(1) # 运行前20个数据
    
    y_true, y_pred = [], []

    # 使用 'w' 模式打开文件，每次运行都会覆盖旧报告
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== Multi-Agent Clinical Diagnosis Benchmark Report ===\n")
        f.write(f"Dataset: {DEV_CSV}\n\n")

        pbar = tqdm(test_df.iterrows(), total=len(test_df), desc="Agent 专家会诊中", unit="patient")

        for _, row in pbar:
            p_id = int(row['Participant_ID'])
            ground_truth = row['PHQ8_Score']
            
            try:
                p_text, au_sum = load_participant_data(DATA_ROOT, p_id)
                
                result = app.invoke({
                    "participant_id": str(p_id),
                    "transcript_text": p_text,
                    "au_summary": au_sum,
                    "retry_count": 0
                })
                
                output = result.get("analysis_output", {})
                pred_score = output.get("total", 0)
                y_true.append(ground_truth)
                y_pred.append(pred_score)

                # --- 构造日志内容 ---
                log_content = []
                log_content.append("█" * 60)
                log_content.append(f"📊 样本 ID: {p_id} | 真实分: {ground_truth} | 预测分: {pred_score}")
                log_content.append(f"📝 核心诊断逻辑: {output.get('rationale', '无')}")
                log_content.append("\n🔍 [维度证据深度拆解]")
                
                t_scores = output.get('individual_scores', {})
                for key, display_name in DISSECT_MAP.items():
                    detail = t_scores.get(key, {})
                    score = detail.get('score', 0)
                    evidence = detail.get('evidence', '未提及相关症状')
                    
                    if score > 0:
                        severity = "轻度" if score == 1 else "中度" if score == 2 else "重度"
                        line = f"   🔴 {display_name}: {score}/3 | 证据: {evidence} | (判定为{severity})"
                    else:
                        line = f"   ⚪ {display_name}: 0 分"
                    log_content.append(line)

                log_content.append(f"\n🛡️ 审计反馈: {result.get('audit_comment', 'N/A')}")
                log_content.append("█" * 60 + "\n\n")

                # --- 同时写入文件和控制台 ---
                full_log = "\n".join(log_content)
                f.write(full_log)
                tqdm.write(full_log)

                pbar.set_postfix({"MAE_Gap": abs(ground_truth - pred_score)})

            except Exception as e:
                error_msg = f"❌ ID {p_id} 运行失败: {e}\n"
                f.write(error_msg)
                tqdm.write(error_msg)

        # 最终汇总
        if y_true:
            mae = mean_absolute_error(y_true, y_pred)
            summary = f"\n✅ 实验完成！最终平均绝对误差 (MAE): {mae:.3f}"
            f.write(summary)
            print(summary)

if __name__ == "__main__":
    main()