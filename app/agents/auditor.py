from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.config import get_openai_key
from app.schemas import AuditReport
from app.prompts import AUDITOR_SYSTEM


class AuditorAgent:
    """
    LLM-based educational auditor with structured critique output.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = get_openai_key()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found (set in .env, env vars, or .streamlit/secrets.toml)"
            )
        self.client = ChatOpenAI(model=model, temperature=0.1, openai_api_key=api_key)

    def audit(self, qb_json: dict, context_snippets: list[dict], targets: dict) -> AuditReport:
        parser = PydanticOutputParser(pydantic_object=AuditReport)
        ctx_lines = []
        for i, snip in enumerate(context_snippets, start=1):
            ctx_lines.append(
                f"[SNIPPET {i}] source={snip.get('source')} page={snip.get('page')} "
                f"source_type={snip.get('source_type','material')}\n{snip.get('text')}\n"
            )
        ctx_text = "\n".join(ctx_lines)

        prompt = f"""
Question Bank JSON:
{qb_json}

Targets:
{targets}

Context snippets:
{ctx_text}
"""

        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", AUDITOR_SYSTEM),
                    ("user", "{prompt}\n\n{format_instructions}"),
                ]
            )
            | self.client
            | parser
        )
        return chain.invoke({"prompt": prompt, "format_instructions": parser.get_format_instructions()})
