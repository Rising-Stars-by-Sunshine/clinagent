import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from utils import load_participant_data
from graph import app

# Path configuration.
DATA_ROOT = r"D:\D\DKU\Dachuang\Dachuang24fall\Dataset" 
DEV_CSV = os.path.join(DATA_ROOT, "data_split_Depression_AVEC2017.csv")
REPORT_FILE = "clinical_diagnosis_report2.txt"

# Display names for PHQ-8 dimensions.
DISSECT_MAP = {
    "1_interest": "Anhedonia",
    "2_mood": "Depressed Mood",
    "3_sleep": "Sleep Disturbance",
    "4_energy": "Fatigue or Low Energy",
    "5_appetite": "Appetite Change",
    "6_self_esteem": "Low Self-Esteem or Worthlessness",
    "7_concentration": "Concentration Difficulty",
    "8_movement": "Psychomotor Agitation or Retardation"
}

def main():
    if not os.path.exists(DEV_CSV):
        print(f"Index file not found: {DEV_CSV}")
        return

    dev_df = pd.read_csv(DEV_CSV)
    test_df = dev_df.head(1)
    
    y_true, y_pred = [], []

    # Overwrite the previous report for each benchmark run.
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== Multi-Agent Clinical Diagnosis Benchmark Report ===\n")
        f.write(f"Dataset: {DEV_CSV}\n\n")

        pbar = tqdm(test_df.iterrows(), total=len(test_df), desc="Running agent consultation", unit="patient")

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

                log_content = []
                log_content.append("=" * 60)
                log_content.append(f"Sample ID: {p_id} | Ground truth: {ground_truth} | Predicted score: {pred_score}")
                log_content.append(f"Diagnostic rationale: {output.get('rationale', 'N/A')}")
                log_content.append("\nItem-level Evidence Analysis")
                
                t_scores = output.get('individual_scores', {})
                for key, display_name in DISSECT_MAP.items():
                    detail = t_scores.get(key, {})
                    score = detail.get('score', 0)
                    evidence = detail.get('evidence', 'No relevant symptoms mentioned.')
                    
                    if score > 0:
                        severity = "mild" if score == 1 else "moderate" if score == 2 else "severe"
                        line = f"   {display_name}: {score}/3 | Evidence: {evidence} | Severity: {severity}"
                    else:
                        line = f"   {display_name}: 0"
                    log_content.append(line)

                log_content.append(f"\nAudit feedback: {result.get('audit_comment', 'N/A')}")
                log_content.append("=" * 60 + "\n\n")

                full_log = "\n".join(log_content)
                f.write(full_log)
                tqdm.write(full_log)

                pbar.set_postfix({"MAE_Gap": abs(ground_truth - pred_score)})

            except Exception as e:
                error_msg = f"ID {p_id} failed: {e}\n"
                f.write(error_msg)
                tqdm.write(error_msg)

        if y_true:
            mae = mean_absolute_error(y_true, y_pred)
            summary = f"\nExperiment completed. Final mean absolute error (MAE): {mae:.3f}"
            f.write(summary)
            print(summary)

if __name__ == "__main__":
    main()