from __future__ import annotations

from rag_pageindex.pageindex.llm.protocol import (
    ContentPart,
    LLMClient,
    Message,
)
from rag_pageindex.pageindex.pdf.renderer import image_part
from rag_pageindex.pageindex.prompts import render


async def answer_with_images(
    llm: LLMClient,
    *,
    question: str,
    images_png: list[bytes],
) -> str:
    """Ask the VLM to answer `question` using `images_png` as the only context.

    Args:
        llm: A multimodal-capable LLMClient (the existing OpenAI-compatible
            client routed through OpenRouter handles image_url parts).
        question: The user's question.
        images_png: Page images as PNG bytes, in the order they should be shown.

    Returns:
        The model's answer, stripped of leading/trailing whitespace.
    """
    prompt = render("vlm_answer.j2", question=question)
    content: list[ContentPart] = [{"type": "text", "text": prompt}]
    content.extend(image_part(b) for b in images_png)
    messages: list[Message] = [{"role": "user", "content": content}]
    response = await llm.acomplete(messages)
    return response.content.strip()
