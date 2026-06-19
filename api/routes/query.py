# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.deps import get_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=3000)
    user_id: str = "api"
    top_k: int = Field(4, ge=1, le=10)
    model_filter: str = ""
    use_memory: bool = False
    validate: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    status: str
    latency: float


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest) -> Dict[str, Any]:
    result = await get_pipeline().answer(
        query=payload.query,
        user_id=payload.user_id,
        k=payload.top_k,
        model_filter=payload.model_filter,
        use_memory=payload.use_memory,
        validate=payload.validate,
    )
    return result
