from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.config import get_openai_key
from app.schemas import QuestionBank, SubjectProfile, TopicPlan
from app.prompts import (
    PLANNER_SYSTEM,
    GENERATOR_SYSTEM,
    SUBJECT_SYSTEM,
    build_planner_prompt,
    build_generator_prompt,
)

DEFAULT_MODEL = "gpt-4o-mini"


class GeneratorAgent:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = get_openai_key()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found (set in .env, env vars, or .streamlit/secrets.toml)"
            )
        self.client = ChatOpenAI(model=model, temperature=0.4, openai_api_key=api_key)
        self.model = model

    def plan(self, topic: str, syllabus_snippets: list[dict]) -> TopicPlan:
        """
        Returns a TopicPlan using STRICT structured parsing (no json.loads).
        This prevents JSONDecodeError when the model outputs extra text.
        """
        prompt = build_planner_prompt(topic, syllabus_snippets)

        parser = PydanticOutputParser(pydantic_object=TopicPlan)
        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", PLANNER_SYSTEM),
                    ("user", "{prompt}\n\n{format_instructions}"),
                ]
            )
            | self.client
            | parser
        )
        return chain.invoke({"prompt": prompt, "format_instructions": parser.get_format_instructions()})

    def generate(
        self,
        course_name: str,
        targets: dict,
        context_snippets: list[dict],
        subject_profile: dict | None = None,
        question_mix: dict | None = None,
        critique: str | None = None,
    ) -> QuestionBank:
        """
        Returns a QuestionBank using STRICT structured parsing.
        """
        prompt = build_generator_prompt(
            course_name,
            targets,
            context_snippets,
            subject_profile=subject_profile,
            question_mix=question_mix,
            critique=critique,
        )

        parser = PydanticOutputParser(pydantic_object=QuestionBank)
        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", GENERATOR_SYSTEM),
                    ("user", "{prompt}\n\n{format_instructions}"),
                ]
            )
            | self.client
            | parser
        )
        return chain.invoke({"prompt": prompt, "format_instructions": parser.get_format_instructions()})

    def classify_subject(self, syllabus_snippets: list[dict]) -> SubjectProfile:
        """
        Returns a SubjectProfile inferred from syllabus/context snippets.
        """
        ctx = "\n\n".join(
            [f"[S] source={s.get('source')} page={s.get('page')}\n{s.get('text')}" for s in syllabus_snippets]
        )
        prompt = f"""
Syllabus/context snippets:
{ctx}

Detect the subject and recommend a question mix.
"""

        parser = PydanticOutputParser(pydantic_object=SubjectProfile)
        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", SUBJECT_SYSTEM),
                    ("user", "{prompt}\n\n{format_instructions}"),
                ]
            )
            | self.client
            | parser
        )
        return chain.invoke({"prompt": prompt, "format_instructions": parser.get_format_instructions()})
