import json, time, os
from defence.classifier_cluster import ClassifierCluster
from dotenv import load_dotenv
from metrics import AttackEvaluator, AttackResult, MetricsCalculator
from defence.heuristic_channel import HeuristicVectorAnalyzer
from defence.shieldgemma import ShieldGemma2BClassifier
from query_agent import QueryAgent
import re
from typing import List

class hydra:
    def __init__(self,
                 huggingface_token: str,
                 cluster_guard: ClassifierCluster,
                 llm_guard: ShieldGemma2BClassifier,
                 heuristic_guard: HeuristicVectorAnalyzer):
        
        self.huggingface_token: str = huggingface_token 
        self.cluster_guard: ClassifierCluster = cluster_guard
        self.llm_guard: ShieldGemma2BClassifier = llm_guard
        self.heuristic_guard: HeuristicVectorAnalyzer = heuristic_guard

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
        and the prompt is classified as unsafe
        '''
        sentences = self.split_sentences(prompt)

         
        for sentence in sentences:
            if not (
                self.cluster_guard.is_safe(sentence)
                and self.heuristic_guard.is_safe(sentence)
                and self.llm_guard.is_safe(sentence)
            ):
                return False  
            
        return True

 