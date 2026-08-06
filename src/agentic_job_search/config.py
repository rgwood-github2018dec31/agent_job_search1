from datetime import datetime

# Model tier constants
MODEL_NAME_HIGH = 'claude-opus-5'  # audit reference standard only (--audit-opus)
MODEL_NAME_MEDIUM = 'claude-sonnet-5'
MODEL_NAME_LOW = 'claude-haiku-4-5'

# Job search parameters
JOB_SEARCH_START_DATE = datetime(2026, 4, 1)
JOB_MAX_AGE_DAYS = 21
JOB_STALE_AGE_DAYS = 30  # hard rule: postings older than this are auto-rated 1
MAX_REFERENCE_JOBS = 20
THINKING_MAX_CHARS = 1000
# Applied-job corpus: only records applied to within this window feed query generation,
# the ideal-role profile, and the already-applied company blocklist. Older PDFs are kept
# on disk but never read.
APPLIED_JOBS_HORIZON_DAYS = 90

# Local/remote LLM MCP tool servers (started via their scripts/start-tool-server.sh)
LLM_LOCAL_MCP_URL = 'http://127.0.0.1:8002/mcp'
LLM_OPENROUTER_MCP_URL = 'http://127.0.0.1:8006/mcp'
LOCAL_MODEL = 'qwen3.6:latest'
OPENROUTER_MODEL = 'z-ai/glm-5.2'

# Stage 1 (discovery) configuration
MAX_SEARCH_QUERIES = 6  # hard cap; every query costs two LinkedIn searches (Canada + EU)
# Turn budget PER QUERY. Each check_and_record_job / queue_candidate call burns a turn, so a
# budget shared across all queries silently starves the later ones (see run_scraper).
SCRAPER_MAX_TURNS_PER_QUERY = 40
QUERY_PROVIDER = 'openrouter'  # 'openrouter' (glm) | 'anthropic'; falls back to Anthropic on failure
COMPANY_MATCH_PROVIDER = 'openrouter'  # 'openrouter' (glm) | 'anthropic'; falls back to Anthropic on failure

# Audit configuration
AUDIT_OPUS_SAMPLE_SIZE = 2  # jobs sampled per un-surfaced pool for --audit-opus

# Stage 2 (evaluation) configuration
RATING_PROVIDER = 'openrouter'  # 'anthropic' | 'openrouter' | 'ollama'
EXTRACTOR_PROVIDER = 'openrouter'  # 'anthropic' (Haiku agentic session) | 'openrouter' (function-calling loop)
EXTRACTOR_OPENROUTER_MAX_ITERATIONS = 10
EXTRACTOR_TOOL_RESULT_MAX_CHARS = 40_000
TRIAGE_ENABLED = True
TRIAGE_THRESHOLD = 1  # skip the rating call when local triage scores <= this (clear low fits)
REFERENCE_SUMMARY_MAX_CHARS = 2500
LLM_JSON_MAX_TOKENS = 3000  # both default models spend tokens on reasoning before the JSON answer
