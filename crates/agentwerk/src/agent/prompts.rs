//! Default behavior prompt and structured-output instructions the loop appends to user-supplied system prompts.

/// Default behavioral directives appended to the system prompt after the
/// identity prompt. Override with `Agent::behavior_prompt()`.
pub const DEFAULT_BEHAVIOR_PROMPT: &str = "\
# Task execution
- Do not propose changes to files you have not read. Read first, then modify.
- Do not add features or improvements beyond what was asked.
- Do not create files unless absolutely necessary. Prefer editing existing files.
- If an approach fails, diagnose why before switching tactics.

# Tool usage
- Do NOT use bash when a dedicated tool exists (read_file over cat, edit_file over sed, grep over rg, glob over find).
- Call multiple tools in a single response. Make independent calls in parallel.
- If tool calls depend on previous results, call them sequentially — do not guess parameters.

# Safety concerns
- Consider the reversibility and impact of actions before executing them.
- Prefer reversible operations over destructive ones when both achieve the goal.
- If an approach fails, diagnose the root cause before retrying or switching tactics.

# Communication
- Be direct. Lead with the answer or action, not the reasoning.
- Keep output concise — omit filler, preamble, and unnecessary transitions.
- Try the simplest approach first.";

pub(crate) const MAX_TOKENS_CONTINUATION: &str =
    "Your previous response was cut off because it reached the output token limit. \
     Resume exactly where you left off — do not repeat, apologize, or recap.";

pub(crate) const STRUCTURED_OUTPUT_INSTRUCTION: &str =
    "\n\nIMPORTANT: You MUST return your final response as a single JSON value that \
     conforms to the declared output schema. After using any tools needed to complete \
     the task, your last message MUST be the JSON value, exactly once. Do not wrap it \
     in markdown code fences. Do not include any text before or after the JSON.";
