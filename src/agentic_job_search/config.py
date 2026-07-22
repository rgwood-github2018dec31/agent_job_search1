from datetime import datetime

# Model tier constants
MODEL_NAME_HIGH = 'claude-opus-4-8'
MODEL_NAME_MEDIUM = 'claude-sonnet-5'
MODEL_NAME_LOW = 'claude-haiku-4-5'

# Job search parameters
JOB_SEARCH_START_DATE = datetime(2026, 4, 1)
JOB_MAX_AGE_DAYS = 21
JOB_STALE_AGE_DAYS = 30  # hard rule: postings older than this are auto-rated 1
MAX_REFERENCE_JOBS = 20
THINKING_MAX_CHARS = 1000

# Local/remote LLM MCP tool servers (started via their scripts/start-tool-server.sh)
LLM_LOCAL_MCP_URL = 'http://127.0.0.1:8002/mcp'
LLM_OPENROUTER_MCP_URL = 'http://127.0.0.1:8006/mcp'
LOCAL_MODEL = 'qwen3.6:latest'
OPENROUTER_MODEL = 'z-ai/glm-5.2'

# Stage 2 (evaluation) configuration
RATING_PROVIDER = 'anthropic'  # 'anthropic' | 'openrouter' | 'ollama'
EXTRACTOR_PROVIDER = 'openrouter'  # 'anthropic' (Haiku agentic session) | 'openrouter' (function-calling loop)
EXTRACTOR_OPENROUTER_MAX_ITERATIONS = 10
EXTRACTOR_TOOL_RESULT_MAX_CHARS = 40_000
TRIAGE_ENABLED = True
TRIAGE_THRESHOLD = 1  # skip the rating call when local triage scores <= this (clear low fits)
REFERENCE_SUMMARY_MAX_CHARS = 2500
LLM_JSON_MAX_TOKENS = 3000  # both default models spend tokens on reasoning before the JSON answer
