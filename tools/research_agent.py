#!/usr/bin/env python3
"""
ResearchAgent implementation that uses the research tool flow.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from agent import Agent
from tools.research import ResearchOrchestrator

logger = logging.getLogger(__name__)

class ResearchAgent(Agent):
    """
    An agent that orchestrates deep research using the ResearchOrchestrator.
    """
    
    def __init__(self, api_key: str, model: str = None, base_url: str = None, enable_logging: bool = True):
        super().__init__(api_key, model, base_url, enable_logging)
        self.scratch_dir = Path("scratch")
        self.scratch_dir.mkdir(exist_ok=True)

    def run_subtask(self, *args, **kwargs):
        """Helper to run the standard agent loop for sub-tasks without recursion."""
        kwargs.setdefault('exclude_tools', []).append('deep_research')
        return super().run(*args, **kwargs)

    def run(self, question: str, max_iterations: int = 30, verbose: bool = True, iteration_callback=None, stream: bool = False, status_prefix: str = ""):
        """Execute the research workflow using the centralized research flow orchestrator."""
        
        # Clear previous artifacts
        for f in self.scratch_dir.glob("subq_*.jsonl"):
            try: f.unlink()
            except: pass
        for f in [self.scratch_dir / "subquestions.jsonl", self.scratch_dir / "subanswers.jsonl"]:
            if f.exists():
                try: f.unlink()
                except: pass

        def on_plan_ready(ordered_questions):
            with open(self.scratch_dir / "subquestions.jsonl", "w", encoding="utf-8") as f:
                for entry in ordered_questions:
                    f.write(json.dumps({"timestamp": datetime.now().isoformat(), "parent_query": question, **entry}) + "\n")
            if iteration_callback: iteration_callback({"type": "plan", "questions": ordered_questions})

        def on_sub_answer_ready(q_num, answer, urls):
            with open(self.scratch_dir / "subanswers.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": datetime.now().isoformat(), "order": q_num, "answer": answer, "urls": urls}) + "\n")
            if iteration_callback: iteration_callback({"type": "sub_answer", "order": q_num, "answer": answer, "urls": urls})

        orchestrator = ResearchOrchestrator(
            call_llm_fn=self._call_llm,
            agent_instance=self,
            on_status_update=iteration_callback,
            on_plan_ready=on_plan_ready,
            on_sub_answer_ready=on_sub_answer_ready
        )
        
        final_answer = orchestrator.run(question)

        try:
            with open(self.scratch_dir / "final_answer.txt", "w", encoding="utf-8") as f:
                f.write(final_answer)
        except Exception as e:
            logger.error(f"Failed to save final answer: {e}")

        if stream:
            def gen(): yield final_answer
            return gen()
        return final_answer
