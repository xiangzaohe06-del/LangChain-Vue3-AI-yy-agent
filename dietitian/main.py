
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from rag.auto_update import configured_auto_updater
from rag.store import knowledge_store

from api import chat, file, knowledge, memory
from common.logger import setup_logging

# 初始化日志配置
setup_logging()

@asynccontextmanager
async def lifespan(app):
    updater = configured_auto_updater(knowledge_store)
    knowledge_store.auto_updater = updater
    if updater:
        updater.start()
    try:
        yield
    finally:
        if updater:
            updater.stop()


app = FastAPI(
    title="Personal Chief API",
    description="私厨",
    version="0.1.0",
    lifespan=lifespan,
)

# 1. 配置跨域资源共享 (CORS)
# 插件开发中，由于请求来自浏览器扩展环境，必须正确配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定插件的 ID 或具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2.挂载路由
app.include_router(chat.router, prefix="/dietitian/api", tags=["对话"])
app.include_router(file.router, prefix="/dietitian/api", tags=["申请上传签名url"])
app.include_router(knowledge.router, prefix="/dietitian/api", tags=["知识库"])
app.include_router(memory.router, prefix="/dietitian/api", tags=["长期记忆"])



if __name__ == "__main__":
    import uvicorn
    # 启动命令：python -m app.main
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
