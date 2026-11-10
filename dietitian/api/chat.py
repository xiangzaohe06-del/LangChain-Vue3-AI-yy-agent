from fastapi import APIRouter, Header, Path
from typing import Annotated
from starlette.responses import StreamingResponse

from agents.dietitian import chat_dietitian, get_messages_history, clear_messages
from schemas.chat import ChatRequest
from agents.memory import USER_ID_PATTERN, user_memory_store
from api.memory import memory_call

router = APIRouter()
OptionalUser = Annotated[str | None, Header(alias='X-User-Id', pattern=USER_ID_PATTERN)]
ThreadPath = Annotated[str, Path(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')]


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """
    流式对话
     Args:
        request (ChatRequest): 请求参数
    """
    await memory_call(user_memory_store.ensure_thread, request.thread_id, request.user_id)
    return StreamingResponse(
        chat_dietitian(request.message, request.image_url, request.thread_id, request.user_id),
        media_type="text/event-stream"
    )



@router.get("/chat/history/{thread_id}")
async def get_chat_history(thread_id: ThreadPath, user_id: OptionalUser = None):
    """
    获取历史消息
     Args:
        thread_id (str): 会话id
    """
    await memory_call(user_memory_store.ensure_thread, thread_id, user_id)
    messages = await memory_call(get_messages_history, thread_id)
    return {"messages": messages}


@router.delete("/chat/history/{thread_id}")
async def clear_chat_history(thread_id: ThreadPath, user_id: OptionalUser = None):
    """
    清空历史消息
    Args:
        thread_id (str): 会话id
    """
    await memory_call(user_memory_store.ensure_thread, thread_id, user_id)
    await memory_call(clear_messages, thread_id)
    return {"success": True}
