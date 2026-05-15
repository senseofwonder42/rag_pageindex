from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

VisualRank = Literal["xsmall", "small", "medium", "large", "xlarge"]


class PageHeading(BaseModel):
    """A section heading that *starts* on a single page."""

    text: str = Field(description="The heading text exactly as it appears on the page.")
    visual_rank: VisualRank = Field(
        description=(
            "Relative visual size of the heading on the page. Use 'xlarge' for the "
            "largest titles in the document, scaling down to 'xsmall' for the "
            "smallest subsection headings."
        ),
    )


class PageInfo(BaseModel):
    """Per-page record returned by the VLM batch call."""

    page_index: int = Field(description="1-based page index for this image.")
    headings: list[PageHeading] = Field(
        default_factory=list,
        description=(
            "Section headings that visually start on this page. Empty list when "
            "the page contains only body content, figures, or page numbers."
        ),
    )
    description: str = Field(
        description="One or two sentences summarising the page's content.",
    )


class PageBatchResponse(BaseModel):
    """Wrapper returned for a batch of N rendered pages."""

    pages: list[PageInfo] = Field(default_factory=list)
