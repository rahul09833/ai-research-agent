import os
import logging
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tavily import TavilyClient

load_dotenv()

# ---------- Logging (this is your "production readiness" story for Q3/Q4) ----------
logging.basicConfig(
    filename="agent.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

tavily_api_key = os.environ.get("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

# ---------- Define tools Gemini can call ----------
web_search_declaration = {
    "name": "web_search_stub",
    "description": (
        "Search the live web for current, factual, or real-world information "
        "(news, facts, rankings, people, current events, prices, etc). "
        "Use this whenever you need information you don't already know with confidence."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"],
    },
}

calculator_declaration = {
    "name": "calculator",
    "description": "Evaluate a basic arithmetic expression, e.g. '12 * 4 + 7'.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}
        },
        "required": ["expression"],
    },
}

tools = types.Tool(function_declarations=[web_search_declaration, calculator_declaration])
config = types.GenerateContentConfig(tools=[tools])


# ---------- Tool implementations (with error handling built in) ----------
def run_web_search_stub(query: str) -> str:
    """
    Real web search using Tavily's free API (1,000 searches/month, no card).
    Name kept as 'run_web_search_stub' so the tool declaration and
    TOOL_FUNCTIONS mapping below don't need to change.
    """
    if tavily_client is None:
        return "Error: TAVILY_API_KEY not set - add it to your .env file to enable real search."
    try:
        response = tavily_client.search(query=query, max_results=4, include_answer=True)
        parts = []
        if response.get("answer"):
            parts.append(f"Direct answer: {response['answer']}")
        for r in response.get("results", []):
            title = r.get("title", "")
            content = r.get("content", "")[:300]
            url = r.get("url", "")
            parts.append(f"- {title}: {content} (source: {url})")
        return "\n".join(parts) if parts else "No search results found."
    except Exception as e:
        logging.error(f"Tavily search failed for query='{query}': {e}")
        return f"Error: search failed - {e}"


def run_calculator(expression: str) -> str:
    try:
        # NOTE: eval() is unsafe for untrusted input in real production code.
        # For a real build, use a safe math parser like the 'ast' module or 'numexpr'.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        logging.error(f"calculator failed for expression='{expression}': {e}")
        return f"Error: could not evaluate expression - {e}"


TOOL_FUNCTIONS = {
    "web_search_stub": run_web_search_stub,
    "calculator": run_calculator,
}


# ---------- Agent loop with retry logic ----------
def run_agent(user_goal: str, max_turns: int = 6, max_retries: int = 2, log_placeholder=None) -> str:
    """
    Run the agent on a user goal.
    log_placeholder is optional - pass a Streamlit st.empty() object to get
    live tool-call updates in the UI. Leave as None for notebook/script use.
    """
    contents = [types.Content(role="user", parts=[types.Part(text=user_goal)])]
    activity_log = []

    for turn in range(max_turns):
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as e:
                error_text = str(e)
                is_rate_limit = "429" in error_text or "RESOURCE_EXHAUSTED" in error_text
                logging.warning(
                    f"API error on attempt {attempt+1} "
                    f"({'rate limit' if is_rate_limit else 'transient error'}): {e}"
                )
                if attempt == max_retries:
                    return f"Agent failed after {max_retries+1} attempts: {e}"
                time.sleep(_extract_retry_delay(error_text) if is_rate_limit else (2 ** attempt))

        candidate = response.candidates[0]

        # Sometimes Gemini returns a candidate with no content at all -
        # usually a safety block, recitation block, or hitting max_tokens
        # before generating anything. Handle that instead of crashing.
        if candidate.content is None or candidate.content.parts is None:
            finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
            logging.warning(f"Empty response from Gemini, finish_reason={finish_reason}")
            return (
                f"The model returned an empty response (reason: {finish_reason}). "
                "Try rephrasing your request."
            )

        function_calls = [
            part.function_call for part in candidate.content.parts
            if part.function_call is not None
        ]

        if not function_calls:
            final_text = "".join(
                part.text for part in candidate.content.parts if part.text
            )
            return final_text or "The model returned an empty response. Try rephrasing your request."

        contents.append(candidate.content)
        function_response_parts = []

        for fc in function_calls:
            tool_name = fc.name
            tool_input = dict(fc.args)
            logging.info(f"Calling tool: {tool_name} with input: {tool_input}")
            activity_log.append(f"🔧 Called `{tool_name}` with {tool_input}")
            if log_placeholder is not None:
                log_placeholder.markdown("\n\n".join(activity_log))

            func = TOOL_FUNCTIONS.get(tool_name)
            if func is None:
                result_text = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    result_text = func(**tool_input)
                except Exception as e:
                    logging.error(f"Tool '{tool_name}' crashed: {e}")
                    result_text = f"Error: tool crashed - {e}"

            function_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result_text},
                )
            )

        contents.append(types.Content(role="user", parts=function_response_parts))

    return "Agent reached max turns without a final answer."


# ---------- Planner + Executor multi-step agent ----------
def _generate_with_retry(prompt_text: str, max_retries: int = 2):
    """
    Call Gemini with retry + backoff. Rate limit errors (429 / RESOURCE_EXHAUSTED)
    get a longer wait than other transient errors, since the free tier resets
    on a per-minute window.
    """
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt_text)])],
            )
        except Exception as e:
            error_text = str(e)
            is_rate_limit = "429" in error_text or "RESOURCE_EXHAUSTED" in error_text

            if attempt == max_retries:
                logging.error(f"Gemini call failed after {max_retries+1} attempts: {e}")
                raise

            wait_seconds = _extract_retry_delay(error_text) if is_rate_limit else (2 ** attempt)
            logging.warning(
                f"Gemini call failed on attempt {attempt+1} "
                f"({'rate limit' if is_rate_limit else 'transient error'}), "
                f"retrying in {wait_seconds}s: {e}"
            )
            time.sleep(wait_seconds)


import re


def _extract_retry_delay(error_text: str, default: int = 15) -> int:
    """
    Gemini's rate-limit errors include the exact wait time it wants
    (e.g. "retryDelay': '56s'" or "Please retry in 56.67s"). Parse it
    out instead of guessing a fixed number, and add a small buffer.
    """
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", error_text)
    if not match:
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_text)
    if match:
        return int(float(match.group(1))) + 3  # small buffer on top
    return default


def _format_history(chat_history: list) -> str:
    """Turn [{'role': 'user'/'assistant', 'content': str}, ...] into readable text."""
    if not chat_history:
        return "(no earlier conversation)"
    lines = []
    for turn in chat_history[-6:]:  # keep last 6 turns to avoid unbounded prompt growth
        speaker = "User" if turn["role"] == "user" else "Agent"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)


def _looks_complex(user_goal: str) -> bool:
    """
    Cheap local check (no API call) for whether a goal likely needs multiple
    steps. Keeps the free-tier rate limit usable by skipping the planning
    call entirely for straightforward single-part questions.
    """
    lowered = user_goal.lower()
    multi_part_signals = [
        " then ", " after that", " and then", ", then", " next ",
        " followed by", " once you", "step 1", "first,", "firstly",
    ]
    has_signal = any(sig in lowered for sig in multi_part_signals)
    many_clauses = user_goal.count(",") >= 2 or user_goal.count(" and ") >= 2
    return has_signal or many_clauses


def generate_plan(user_goal: str, chat_history: list = None, max_steps: int = 5) -> list:
    """
    Ask Gemini to break the goal into a short ordered list of sub-steps.
    A simple, single-part ask naturally comes back as a 1-step plan -
    there's no separate "simple mode", the planner just adapts.
    Returns a plain Python list of step strings (falls back to a single
    step containing the original goal if parsing fails).
    """
    history_text = _format_history(chat_history)
    planning_prompt = (
        "You are planning how to address a user's goal, possibly using earlier "
        "conversation as context.\n\n"
        f"Earlier conversation:\n{history_text}\n\n"
        f"New goal: {user_goal}\n\n"
        f"Break this goal into the smallest ordered list of concrete sub-steps needed "
        f"(max {max_steps} steps). If it's simple enough to do in one step, return a "
        "list with just one item. Respond with ONLY a JSON array of strings, nothing "
        "else, no markdown fences."
    )
    try:
        response = _generate_with_retry(planning_prompt)
        raw_text = "".join(
            part.text for part in response.candidates[0].content.parts if part.text
        ).strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        steps = json.loads(raw_text)
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps) and steps:
            logging.info(f"Plan generated: {steps}")
            return steps[:max_steps]
    except Exception as e:
        logging.warning(f"Plan generation failed, falling back to single step: {e}")

    return [user_goal]


def synthesize_final_answer(user_goal: str, step_results: list, chat_history: list = None) -> str:
    """Combine all step results (and any earlier conversation) into one coherent answer."""
    history_text = _format_history(chat_history)
    summary_input = "\n\n".join(
        f"Step {i+1}: {step}\nResult: {result}"
        for i, (step, result) in enumerate(step_results)
    )
    synthesis_prompt = (
        f"Earlier conversation:\n{history_text}\n\n"
        f"User's new goal: {user_goal}\n\n"
        f"Steps taken to address it:\n\n{summary_input}\n\n"
        "Write a single, clear, conversational final answer to the user's new goal, "
        "using these results and staying consistent with the earlier conversation. "
        "Do not repeat the step-by-step breakdown, just give the final answer."
    )
    try:
        response = _generate_with_retry(synthesis_prompt)
        return "".join(
            part.text for part in response.candidates[0].content.parts if part.text
        )
    except Exception as e:
        logging.error(f"Synthesis failed: {e}")
        return "Could not synthesize final answer: " + str(e)


def run_smart_agent(user_goal: str, chat_history: list = None, log_placeholder=None) -> str:
    """
    Single unified entry point. Always plans first - a simple ask naturally
    comes back as a 1-step plan, so there's no separate "simple" vs "planner"
    mode to choose between. Supports optional conversation history for
    multi-turn context.

    log_placeholder is optional - pass a Streamlit st.empty() object for
    live progress updates in the UI. Leave as None for notebook/script use.
    """
    activity_log = []

    def update_log(line: str):
        activity_log.append(line)
        if log_placeholder is not None:
            log_placeholder.markdown("\n\n".join(activity_log))

    if _looks_complex(user_goal):
        update_log("🧭 Multi-part request - thinking about how to approach it...")
        steps = generate_plan(user_goal, chat_history=chat_history)
    else:
        steps = [user_goal]  # skip the planning API call entirely - saves a request

    if len(steps) > 1:
        update_log("📋 Plan:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)))
    else:
        update_log("➡️ Simple enough to handle directly.")

    step_results = []
    for i, step in enumerate(steps):
        if len(steps) > 1:
            update_log(f"▶️ Executing step {i+1}/{len(steps)}: {step}")
        result = run_agent(step, log_placeholder=None)
        step_results.append((step, result))

    if len(steps) == 1:
        # No need to spend an extra API call "synthesizing" a single result
        return step_results[0][1]

    if len(steps) > 1:
        update_log("🧩 Combining results into a final answer...")
    final_answer = synthesize_final_answer(user_goal, step_results, chat_history=chat_history)
    return final_answer