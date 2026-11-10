import logging

from fastapi import APIRouter, Query, HTTPException

from rag.store import knowledge_store

router = APIRouter()
logger = logging.getLogger(__name__)


def storage_call(function, *args):
    try:
        return function(*args)
    except Exception as error:
        logger.warning("Knowledge storage unavailable (%s)", type(error).__name__)
        raise HTTPException(status_code=503, detail="知识库存储暂不可用，请检查配置和服务连接") from None


@router.get("/knowledge/status")
def knowledge_status():
    return storage_call(knowledge_store.status)


@router.get("/knowledge/search")
def knowledge_search(q: str = Query(min_length=1, max_length=1000),
                     top_k: int = Query(default=4, ge=1, le=8)):
    return storage_call(knowledge_store.search, q, top_k)
