from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TocDetectedResponse(BaseModel):
    toc_detected: Literal["yes", "no"]


class PageIndexInTocResponse(BaseModel):
    page_index_given_in_toc: Literal["yes", "no"]


class CompletionCheckResponse(BaseModel):
    completed: Literal["yes", "no"]


class TitleAppearanceResponse(BaseModel):
    answer: Literal["yes", "no"]
    at_start: Literal["yes", "no"] = "no"


class PhysicalIndexResponse(BaseModel):
    physical_index: str | None = Field(
        default=None,
        description='Physical page index in "<physical_index_X>" format, or null',
    )
    confidence: Literal["high", "low"] = Field(
        default="low",
        description='"high" only if the section heading appears verbatim or '
        'nearly verbatim at the top of exactly one page; otherwise "low".',
    )


class TocEntry(BaseModel):
    structure: str | None = Field(
        default=None,
        description='Hierarchy index in "x.x.x" format, or null',
    )
    title: str
    page: int | None = None


class TocTransformResponse(BaseModel):
    table_of_contents: list[TocEntry]


class TocEntryWithPhysicalIndex(BaseModel):
    structure: str | None = None
    title: str
    physical_index: str | None = Field(
        default=None,
        description='Physical page index in "<physical_index_X>" format, or null',
    )


class TocIndexResponse(BaseModel):
    """Wrapper for toc_index_extractor results."""

    items: list[TocEntryWithPhysicalIndex] = Field(default_factory=list)


class TocEntryWithPageNumber(BaseModel):
    structure: str | None = None
    title: str
    start: Literal["yes", "no"] = "no"
    physical_index: str | None = Field(
        default=None,
        description='Physical page index in "<physical_index_X>" format, or null',
    )


class TocPageNumberResponse(BaseModel):
    """Wrapper for add_page_number_to_toc results."""

    items: list[TocEntryWithPageNumber] = Field(default_factory=list)


class TocGeneratedEntry(BaseModel):
    structure: str = Field(description='Hierarchy index in "x.x.x" format')
    title: str = Field(description="Original title extracted from text")
    physical_index: str = Field(
        description='Physical page index in "<physical_index_X>" format',
    )


class TocGeneratedResponse(BaseModel):
    """Wrapper for generate_toc_init / generate_toc_continue results."""

    items: list[TocGeneratedEntry] = Field(default_factory=list)
