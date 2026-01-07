import json
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class ResearchOrchestrator:
    """
    Orchestrates a multi-step research flow:
    1. Decomposition of a query into sub-questions.
    2. Ranking and ordering of sub-questions.
    3. Sequential answering of sub-questions with context sharing.
    4. Intelligent source attribution for each sub-question.
    5. Final synthesis of a comprehensive report.
    """
    
    def __init__(
        self, 
        call_llm_fn: Callable, 
        agent_instance: Any,
        on_status_update: Optional[Callable[[str], None]] = None,
        on_plan_ready: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        on_sub_answer_ready: Optional[Callable[[int, str, List[str]], None]] = None
    ):
        self.call_llm_fn = call_llm_fn
        self.agent_instance = agent_instance
        self.on_status_update = on_status_update
        self.on_plan_ready = on_plan_ready
        self.on_sub_answer_ready = on_sub_answer_ready
        self.sub_answers = []

    def _update_status(self, msg: str):
        if self.on_status_update:
            self.on_status_update(msg)

    def run(self, query: str) -> str:
        """Executes the full research orchestration logic."""
        
        # 1. Decompose
        self._update_status("Decomposing query into sub-questions...")
        sub_questions = self._decompose_query(query)
        
        # 2. Rank and Order
        self._update_status("Ranking and ordering research tasks...")
        ordered_questions_meta = self._rank_and_order_subquestions(query, sub_questions)
        
        if self.on_plan_ready:
            self.on_plan_ready(ordered_questions_meta)
        
        total_q = len(ordered_questions_meta)
        
        # 3. Process sub-questions
        for idx, meta in enumerate(ordered_questions_meta, 1):
            sub_q = meta["question"]
            q_num = meta["order"]
            status_prefix = f"[Sub-question {idx}/{total_q}]"
            
            self._update_status(f"{status_prefix} Researching: {sub_q}")
            
            # State for this specific sub-question
            read_urls = []
            search_results = []

            def interceptor(name, args, result, runtime):
                if name == "read_url":
                    u = args.get("url")
                    if u and u not in read_urls: read_urls.append(u)
                if name == "web_search":
                    try:
                        res_list = json.loads(result)
                        if isinstance(res_list, list):
                            for r in res_list:
                                if isinstance(r, dict) and r.get("href"):
                                    search_results.append(r)
                    except: pass

            self.agent_instance.on_tool_call = interceptor

            # Build context and run
            context = "\n".join([f"Q: {sq}\nA: {sa}" for sq, sa in self.sub_answers])
            prompt = f"Previous Findings:\n{context}\n\nTarget Question: {sub_q}" if context else sub_q
            
            runner = getattr(self.agent_instance, "run_subtask", self.agent_instance.run)
            sub_answer = runner(
                prompt, max_iterations=15, verbose=True, 
                iteration_callback=lambda it: self._update_status(f"{status_prefix} Iteration {it+1}..."),
                status_prefix=status_prefix
            )
            
            # Attribution
            self._update_status(f"{status_prefix} Attributing sources...")
            cited_urls = self._attribute_sources(sub_q, sub_answer, search_results)
            all_sources = list(set(cited_urls + read_urls))
            
            self.sub_answers.append((sub_q, sub_answer))
            if self.on_sub_answer_ready:
                self.on_sub_answer_ready(q_num, sub_answer, all_sources)

        self.agent_instance.on_tool_call = None

        # 4. Synthesize
        self._update_status("Synthesizing final research report...")
        return self._synthesize(query)

    def _decompose_query(self, query: str) -> List[str]:
        prompt = f"Decompose this query into a JSON list of 3-5 sub-questions: {query}. Return ONLY JSON."
        try:
            result = self.call_llm_fn([{"role": "user", "content": prompt}], False)
            content = result["choices"][0]["message"]["content"].strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            parsed = json.loads(content)
            return parsed if isinstance(parsed, list) else [query]
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return [query]

    def _rank_and_order_subquestions(self, query: str, sub_questions: List[str]) -> List[Dict[str, Any]]:
        prompt = f"Rank and order these sub-questions for the query '{query}': {json.dumps(sub_questions)}. Return a JSON list of objects with 'question', 'rank', 'order', 'reasoning'."
        try:
            result = self.call_llm_fn([{"role": "user", "content": prompt}], False)
            content = result["choices"][0]["message"]["content"].strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            ordered = json.loads(content)
            ordered.sort(key=lambda x: x.get("order", 99))
            return ordered
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return [{"question": q, "rank": 3, "order": i+1, "reasoning": "Fallback"} for i, q in enumerate(sub_questions)]

    def _attribute_sources(self, sub_q: str, answer: str, search_results: List[Dict[str, Any]]) -> List[str]:
        if not search_results: return []
        context = "".join([f"[{i+1}] {res.get('title')} ({res.get('href')})\n" for i, res in enumerate(search_results)])
        prompt = f"Identify used results (JSON list of numbers [1, 3]):\nQ: {sub_q}\nA: {answer}\nResults:\n{context}"
        try:
            result = self.call_llm_fn([{"role": "user", "content": prompt}], False)
            content = result["choices"][0]["message"]["content"].strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            indices = json.loads(content)
            cited = []
            for i in indices:
                idx = int(i) - 1
                if 0 <= idx < len(search_results):
                    url = search_results[idx].get("href")
                    if url and url not in cited: cited.append(url)
            return cited
        except: return []

    def _synthesize(self, query: str) -> str:
        findings = "\n".join([f"Q: {sq}\nA: {sa}" for sq, sa in self.sub_answers])
        prompt = f"Synthesize report for: {query}\n\nFindings:\n{findings}\n\nComprehensive Report (Markdown):"
        result = self.call_llm_fn([{"role": "user", "content": prompt}], False)
        answer = result["choices"][0]["message"]["content"].strip()
        for p in ["Comprehensive Report:", "**Comprehensive Report**"]:
            if answer.startswith(p): answer = answer[len(p):].strip()
        return answer

def perform_research_flow(query: str, call_llm_fn: Callable, agent_instance: Any, **kwargs) -> str:
    """Legacy wrapper for the class-based orchestrator."""
    orchestrator = ResearchOrchestrator(call_llm_fn, agent_instance, **kwargs)
    return orchestrator.run(query)