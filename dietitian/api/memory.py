import asyncio
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException

from agents.memory import (MemoryConflict, MemoryDenied, SaveMemory, USER_ID_PATTERN, user_memory_store)

router = APIRouter()
UserHeader = Annotated[str, Header(alias='X-User-Id', pattern=USER_ID_PATTERN)]


async def memory_call(function, *args, **kwargs):
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except MemoryConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except MemoryDenied as error:
        raise HTTPException(status_code=403, detail=str(error)) from None
    except Exception:
        raise HTTPException(status_code=503, detail='长期记忆暂不可用，请稍后重试。') from None


@router.get('/memory/profile')
async def get_memory(user_id: UserHeader):
    return await memory_call(user_memory_store.get, user_id)


@router.put('/memory/profile')
async def save_memory(request: SaveMemory, user_id: UserHeader):
    return await memory_call(user_memory_store.save, user_id, request.facts.model_dump(), revision=request.revision)


@router.delete('/memory/profile')
async def clear_memory(user_id: UserHeader):
    return await memory_call(user_memory_store.forget, user_id)
