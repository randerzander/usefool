import json
import asyncio
import time
import os
import yaml
import sys
from datetime import datetime
from openai import OpenAI
from pathlib import Path
from agent import Agent
from colorama import Fore, Style
from utils import is_localhost, get_model_for_api

# Load config
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Setup query logs directory
DATA_DIR = Path(__file__).parent / "data"
QUERY_LOGS_DIR = DATA_DIR / "query_logs"
QUERY_LOGS_DIR.mkdir(parents=True, exist_ok=True)

def judge_answer(question: str, expected_answer: str, agent_response: str) -> dict:
    """Use LLM to judge if the agent's response contains the expected answer."""
    
    # Get API settings from config
    api_key_env = CONFIG.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env, "sk-fake")
    base_url = CONFIG.get("base_url", "http://localhost:8080/v1")
    
    # Strip /chat/completions from base_url if present (OpenAI client adds it)
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Use shared utility to get model
    model = get_model_for_api(base_url, api_key, CONFIG.get("default_model", "gpt-3.5-turbo"))
    
    judge_prompt = f"""
EXPECTED: {expected_answer}
ACTUAL: {agent_response}

Do NOT assess the truth of ACTUAL or Expected. Only assess whether ACTUAL contains the information in EXPECTED.

You may provide reasoning, but you MUST end your response with a final decision on its own line.
Your final line must be ONLY one of: PASS or FAIL

if ACTUAL contains EXPECTED, your final line should be: PASS
if ACTUAL does not contain EXPECTED, your final line should be: FAIL
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    
    judgment_text = response.choices[0].message.content.strip()
    
    # Get the last non-empty line for the final decision
    lines = [line.strip() for line in judgment_text.split('\n') if line.strip()]
    final_decision = lines[-1] if lines else ""
    is_pass = final_decision.upper() == 'PASS'
    
    # Remove the final PASS/FAIL line from the judgment text for display
    reason_lines = lines[:-1] if lines and final_decision.upper() in ['PASS', 'FAIL'] else lines
    reason_text = '\n'.join(reason_lines)
    
    return {
        "passed": is_pass,
        "judgment": reason_text
    }

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run agent tests')
    parser.add_argument('-qid', type=int, help='Run only the question with this QID')
    parser.add_argument('-iterations', type=int, default=1, help='Number of times to run the entire eval (default: 1)')
    args = parser.parse_args()
    
    overall_start = time.time()
    all_iterations_results = []
    
    # Create results file upfront and initialize it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for iteration in range(args.iterations):
        if args.iterations > 1:
            print(f"\n{'#'*60}")
            print(f"# ITERATION {iteration + 1} of {args.iterations}")
            print(f"{'#'*60}\n")
        
        start_time = time.time()
        
        # Initialize agent
        api_key_env = CONFIG.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        base_url = CONFIG.get("base_url", "http://localhost:8080/v1")
        
        # Pass model=None for localhost to trigger auto-detection in Agent
        # For remote APIs, get from config
        model = None if is_localhost(base_url) else CONFIG.get("default_model", "gpt-3.5-turbo")
        
        # Don't strip base_url for agent - it expects the full path
        agent = Agent(api_key=api_key, model=model, base_url=base_url)
        
        # Create results file on first iteration
        if iteration == 0:
            model_safe = agent.model.replace('/', '_').replace('\\', '_').replace('.gguf', '')
            if args.iterations > 1:
                results_file = Path.cwd() / f"{model_safe}_{args.iterations}iterations_{timestamp}.jsonl"
            else:
                results_file = Path.cwd() / f"{model_safe}_{timestamp}.jsonl"
            # Create empty file
            results_file.touch()
            print(f"📝 Writing results to: {results_file.name}\n")
        
        with open("data/qa.jsonl", "r") as f:
            qa_pairs = [json.loads(line) for line in f if line.strip()]
        
        # Filter out questions with blank answers
        qa_pairs = [qa for qa in qa_pairs if qa.get("answer", "").strip()]
        
        # Filter by qid if specified
        if args.qid:
            qa_pairs = [qa for qa in qa_pairs if qa.get("qid") == args.qid]
            if not qa_pairs:
                print(f"No question found with qid={args.qid}")
                return
        
        print(f"\nRunning evaluation on {len(qa_pairs)} questions with answers...")
        
        results = []
        passed_count = 0
        failed_count = 0
        
        for idx, qa in enumerate(qa_pairs, 1):
            question = qa["question"]
            expected = qa["answer"]
            qid = qa.get("qid", "?")
        
        # Print question and expected answer before agent runs
        print(f"\n{'='*60}")
        print(f"Question {idx}/{len(qa_pairs)} | QID: {qid}")
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        print(f"{'='*60}")
        
        q_start_time = time.time()
        agent_response = agent.run(question, verbose=False)
        q_elapsed = time.time() - q_start_time
        
        judgment = judge_answer(question, expected, agent_response)
        
        if judgment['passed']:
            passed_count += 1
            result_symbol = f"{Fore.GREEN}✓{Style.RESET_ALL}"
        else:
            failed_count += 1
            result_symbol = f"{Fore.RED}✗{Style.RESET_ALL}"
        
        # Print result
        print(f"\n{result_symbol} Result | {q_elapsed:.2f}s")
        print(f"A: {agent_response[:200]}{'...' if len(agent_response) > 200 else ''}")
        if not judgment['passed'] and judgment['judgment']:
            print(f"Reason: {judgment['judgment']}")
        print(f"{'='*60}")
        
        # Log the execution chain to query_logs (like discord_bot does)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = QUERY_LOGS_DIR / f"eval_qid{qid}_{timestamp}.json"
        
        # Get tracking data from agent
        tracking = agent.get_tracking_data()
        
        log_data = {
            "qid": qid,
            "username": "eval",
            "timestamp": datetime.now().isoformat(),
            "user_query": question,
            "expected_answer": expected,
            "final_response": agent_response,
            "passed": judgment["passed"],
            "judgment": judgment["judgment"],
            "execution_time": q_elapsed,
            "call_sequence": tracking.get("call_sequence", []),
            "token_stats": tracking.get("token_stats", {})
        }
        
        with open(log_filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        result_entry = {
            "qid": qid,
            "question": question,
            "expected": expected,
            "response": agent_response,
            "passed": judgment["passed"],
            "judgment": judgment["judgment"],
            "time": q_elapsed,
            "iteration": iteration + 1
        }
        
        results.append(result_entry)
        all_iterations_results.append(result_entry)
        
        # Append to results file immediately
        with open(results_file, 'a') as f:
            f.write(json.dumps(result_entry) + '\n')
        
        total_time = time.time() - start_time
        
        # Clear any remaining status line
        sys.stdout.write("\x1b[2K\r")
        sys.stdout.flush()
        
        print(f"\n{'='*60}")
        print(f"SUMMARY (Iteration {iteration + 1}): ", end="")
        
        # Colored summary
        if passed_count > 0:
            print(f"{Fore.GREEN}{passed_count} passed{Style.RESET_ALL}", end="")
        else:
            print(f"{passed_count} passed", end="")
        
        print(" / ", end="")
        
        if failed_count > 0:
            print(f"{Fore.RED}{failed_count} failed{Style.RESET_ALL}", end="")
        else:
            print(f"{failed_count} failed", end="")
        
        print(f" / {len(results)} total")
        
        print(f"Total Runtime: {total_time:.2f}s")
        if len(results) > 0:
            print(f"Average Time per Question: {total_time/len(results):.2f}s")
        print(f"{'='*60}")
    
    # After all iterations, show final summary
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*60}")
    if args.iterations > 1:
        # Calculate aggregate stats
        total_questions = len(all_iterations_results)
        total_passed = sum(1 for r in all_iterations_results if r['passed'])
        total_failed = total_questions - total_passed
        
        print(f"OVERALL SUMMARY ({args.iterations} iterations):")
        print(f"  Total questions: {total_questions}")
        print(f"  {Fore.GREEN}Passed: {total_passed}{Style.RESET_ALL}")
        print(f"  {Fore.RED}Failed: {total_failed}{Style.RESET_ALL}")
        print(f"  Pass rate: {total_passed/total_questions*100:.1f}%")
        print(f"  Total runtime: {overall_time:.2f}s")
    
    print(f"✅ Results saved to: {results_file.name}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
