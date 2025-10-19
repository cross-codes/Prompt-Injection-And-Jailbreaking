import re
from typing import List
from classifier_cluster import ClassifierCluster
from heuristic_channel import HeuristicVectorAnalyzer
from shieldgemma import ShieldGemma2BClassifier


class SubPromptDefence:

    def __init__(self,
                 huggingface_token: str,
                 heuristic_window_size: int = 3,  
                 heuristic_threshold: int = 3):   
        
         
        self.huggingface_token: str = huggingface_token 
        self.cluster_guard: ClassifierCluster = ClassifierCluster()
        self.heuristic_guard: HeuristicVectorAnalyzer = HeuristicVectorAnalyzer(
            heuristic_window_size, 
            heuristic_threshold
        )
        self.llm_guard: ShieldGemma2BClassifier = ShieldGemma2BClassifier(huggingface_token)

    def split_sentences(self, prompt:str) -> List[str]:
        '''
        Splits a prompt into individual sentences based on special characters.
        '''
        sentences = re.split(r'[.!?]+', prompt)
        sentences = [s.strip() for s in sentences if s.strip()] 
        return sentences
    
    def is_safe(self, prompt: str) -> bool:
        '''
        Sentences are passed through the guardrails.
        If a single sentence is classified as unsafe then evaluation is stopped
        and the prompt is classified as unsafe entirely.
        '''
        sentences = self.split_sentences(prompt)

        if not(self.heuristic_guard.is_safe(prompt)):
            return False
        else:
            for sentence in sentences:
                if not (
                    self.cluster_guard.is_safe(sentence)
                    and self.llm_guard.is_safe(sentence)
                    ):
                    return False

        return True
