"""
generation/generator.py
Calls Groq API for answer generation.
Groq runs Llama 3.1, Mixtral — free tier, 6000 req/day.
Uses AsyncGroq so the FastAPI event loop is never blocked.
"""
import logging
from groq import AsyncGroq
from models import ChunkResult
from config import settings
from langsmith import traceable  
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import APIStatusError, APITimeoutError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical decision support assistant for healthcare professionals.

Rules you must follow without exception:
1. Answer ONLY using the context provided below. Never use outside knowledge.
2. If the context does not contain enough information, respond with:
   "The provided documents do not contain sufficient information to answer this question."
3. Cite source numbers [1], [2] etc. for every factual claim.
4. Use clinical terminology where appropriate. Be precise, not conversational.
5. Never speculate, infer, or guess beyond what the context explicitly states.

This system operates in a HIPAA-compliant environment."""


def _build_context_block(chunks: list[ChunkResult]) -> str:
    """
    Format chunks into numbered context block with full location metadata.
    Clinicians need exact page + section to verify sources in original documents.
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk.source.replace("\\", "/").split("/")[-1]

        location_parts = [f"Source: {filename}"]
        if chunk.page_number and chunk.page_number > 0:
            location_parts.append(f"Page {chunk.page_number}")
        if chunk.section_header:
            location_parts.append(f"Section: {chunk.section_header}")

        location = " | ".join(location_parts)
        lines.append(f"[{i}] {location}\n{chunk.text.strip()}")
    return "\n\n".join(lines)


class Generator:

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model  = settings.llm_model
        logger.info(f"Generator initialized: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIStatusError, APITimeoutError)),
        reraise=True,
    )
    async def _call_llm(self, messages: list) -> object:
        """
        Call Groq with exponential backoff retry.
        Retries on rate limits (429) and timeouts.
        Fails fast on auth errors (401) — no point retrying bad credentials.
        Max 3 attempts: 0s, 2s, 4s wait between retries.
        """
        return await self.client.chat.completions.create(
            model      = self.model,
            messages   = messages,
            temperature= 0.1,
            max_tokens = 600,
        )

    async def generate(self, query, chunks, patient_id=None):
        if not chunks:
            logger.warning(f"No chunks for query: '{query[:50]}'")
            return {
                "answer":     "No relevant context found in the knowledge base.",
                "sources":    [],
                "model_used": self.model,
                "tokens_used": 0,
            }

        context_block = _build_context_block(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context from clinical documents:\n\n"
                    f"{context_block}\n\n"
                    f"Question: {query}\n\n"
                    f"Answer strictly from the context. "
                    f"Cite source numbers [1], [2] for each claim."
                ),
            },
        ]

        response    = await self._call_llm(messages)
        answer      = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens

        sources = list({
            chunk.source.replace("\\", "/").split("/")[-1]
            for chunk in chunks
        })

        logger.info(
            f"Generated: {len(answer)} chars | "
            f"{len(sources)} source(s) | {tokens_used} tokens"
        )

        return {
            "answer":      answer,
            "sources":     sources,
            "model_used":  self.model,
            "tokens_used": tokens_used,
        }



#This is the old setup as it didn't had the retry logic for production

'''
class Generator:
    """
    Generates clinical answers from ranked chunks using Groq.
    AsyncGroq is natively async — no asyncio.to_thread() needed.
    """

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
        logger.info(f"Generator initialized: {self.model}")


    """async def generate(
        self,
        query: str,
        chunks: list[ChunkResult],
        patient_id: str | None = None,
    ) -> dict:"""
    

    @traceable(name="clinical_llm_generator", run_type="llm")
    async def generate(self, query, chunks, patient_id=None):
        """
        Build clinical prompt and generate answer from Groq.
        Returns answer, sources, model used, tokens consumed.
        """
        if not chunks:
            logger.warning(f"No chunks for query: '{query[:50]}'")
            return {
                "answer": "No relevant context found in the knowledge base.",
                "sources": [],
                "model_used": self.model,
                "tokens_used": 0,
            }

        context_block = _build_context_block(chunks)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context from clinical documents:\n\n"
                        f"{context_block}\n\n"
                        f"Question: {query}\n\n"
                        f"Answer strictly from the context. "
                        f"Cite source numbers [1], [2] for each claim."
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=600,
        )

        answer = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens

        seen = set()
        sources = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.source.replace("\\", "/").split("/")[-1]
            page     = chunk.page_number if chunk.page_number and chunk.page_number > 0 else "?"
            ref      = f"[{i}] {filename} — Page {page}"
            if ref not in seen:
                seen.add(ref)
                sources.append(ref)
        

        logger.info(
            f"Generated: {len(answer)} chars | "
            f"{len(sources)} source(s) | "
            f"{tokens_used} tokens"
        )

        return {
            "answer": answer,
            "sources": sources,
            "model_used": self.model,
            "tokens_used": tokens_used,
        }
'''
