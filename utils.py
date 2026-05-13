import pandas as pd
import os

def load_participant_data(data_root, p_id):
    """
    Load and preprocess multimodal data for one participant.
    """
    # 1. Process transcript data.
    t_filename = f"{p_id}_TRANSCRIPT.csv"
    t_path = os.path.join(data_root, t_filename)
    
    if not os.path.exists(t_path):
        return "", f"Error: transcript file not found: {t_path}"

    try:
        # DAIC-WOZ transcript files are commonly tab-separated.
        df_t = pd.read_csv(t_path, sep='\t').fillna("")
        if 'speaker' not in df_t.columns:
            df_t = pd.read_csv(t_path).fillna("")
            
        p_text = " ".join(df_t[df_t['speaker'] == 'Participant']['value'].astype(str).tolist())
    except Exception as e:
        p_text = f"Error parsing transcript: {str(e)}"
    
    # 2. Process AU visual features.
    au_filename = f"{p_id}_CLNF_AUs.csv"
    au_path = os.path.join(data_root, au_filename)
    
    if not os.path.exists(au_path):
        au_summary = "Visual feature data is missing."
    else:
        try:
            df_au = pd.read_csv(au_path)
            # Exclude frames where face tracking failed.
            valid_au = df_au[df_au['success'] == 1]
            
            # Use compact AU summaries to reduce prompt length.
            au12_mean = valid_au['AU12_r'].mean() if 'AU12_r' in valid_au else 0
            au04_freq = (valid_au['AU04_c'] == 1).mean() if 'AU04_c' in valid_au else 0
            
            au_summary = f"AU12 smile intensity mean: {au12_mean:.2f}, AU04 brow furrow frequency: {au04_freq:.2%}"
        except Exception as e:
            au_summary = f"Error parsing AU data: {str(e)}"
    return p_text, au_summary
    