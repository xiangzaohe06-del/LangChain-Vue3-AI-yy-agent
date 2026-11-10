from typing import List

from pydantic import BaseModel, Field
from agents.memory import UserId

# ---  数据模型 ---
class ChatRequest(BaseModel):
    # 消息内容
    message: str = Field(max_length=8000)
    # 图片地址
    image_url: List[str] = Field(default_factory=list, max_length=4)
    # 会话id
    thread_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    user_id: UserId | None = None
