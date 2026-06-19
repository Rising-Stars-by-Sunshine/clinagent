from typing import TypedDict, Dict, List

class MentalState(TypedDict):
    participant_id: str
    transcript_text: str    
    au_summary: str        
    
    perception_log: str     
    knowledge_base: str   
    
    analysis_output: Dict  
    audit_comment: str    
    retry_count: int        
    