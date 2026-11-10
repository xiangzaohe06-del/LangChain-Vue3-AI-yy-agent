"""Explicit, user-scoped dietary memory, independent from conversation checkpoints."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

USER_ID_PATTERN = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$'
UserId = Annotated[str, Field(pattern=USER_ID_PATTERN)]
Item = Annotated[str, Field(min_length=1, max_length=60)]
MemoryField = Literal['allergies', 'avoids', 'likes', 'diet', 'goals', 'budget']


class MemoryFacts(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    allergies: list[Item] = Field(default_factory=list, max_length=12)
    avoids: list[Item] = Field(default_factory=list, max_length=12)
    likes: list[Item] = Field(default_factory=list, max_length=12)
    diet: str = Field(default='', max_length=80)
    goals: list[Item] = Field(default_factory=list, max_length=12)
    budget: str = Field(default='', max_length=80)

    @field_validator('allergies', 'avoids', 'likes', 'goals')
    @classmethod
    def unique(cls, values):
        return list(dict.fromkeys(values))


class MemoryPatch(BaseModel):
    """Only supply explicitly mentioned fields; omitted fields remain untouched."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    allergies: list[Item] | None = Field(default=None, max_length=12)
    avoids: list[Item] | None = Field(default=None, max_length=12)
    likes: list[Item] | None = Field(default=None, max_length=12)
    diet: str | None = Field(default=None, max_length=80)
    goals: list[Item] | None = Field(default=None, max_length=12)
    budget: str | None = Field(default=None, max_length=80)


class SaveMemory(BaseModel):
    model_config = ConfigDict(extra='forbid')
    facts: MemoryFacts
    revision: int = Field(ge=0)


class MemoryConflict(RuntimeError):
    pass


class MemoryDenied(ValueError):
    pass


class UserMemoryStore:
    def __init__(self, collection=None, owners=None):
        self._collection, self._owners = collection, owners
        self._client = None

    @property
    def collection(self):
        if self._collection is None:
            self._client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017'),
                                       serverSelectionTimeoutMS=3000)
            database = self._client[os.getenv('USER_MEMORY_DATABASE', 'dietitian')]
            self._collection = database['user_memories']
            self._owners = database['conversation_owners']
        return self._collection

    def get(self, user_id):
        self._validate_user(user_id)
        record = self.collection.find_one({'_id': user_id})
        return self._public(record)

    @staticmethod
    def _validate_user(user_id):
        if not isinstance(user_id, str) or not re.fullmatch(USER_ID_PATTERN, user_id):
            raise MemoryDenied('Invalid user identity')

    @staticmethod
    def _public(record):
        if not record:
            return {'facts': MemoryFacts().model_dump(), 'revision': 0, 'updated_at': None}
        return {'facts': MemoryFacts.model_validate(record.get('facts', {})).model_dump(),
                'revision': record.get('revision', 0), 'updated_at': record.get('updated_at')}

    def save(self, user_id, facts, *, revision, source='user_form'):
        self._validate_user(user_id)
        facts = MemoryFacts.model_validate(facts).model_dump()
        now = datetime.now(timezone.utc).isoformat()
        try:
            record = self.collection.find_one_and_update(
                {'_id': user_id, 'revision': revision} if revision else {'_id': user_id, 'revision': {'$exists': False}},
                {'$set': {'facts': facts, 'updated_at': now, 'source': source},
                 '$setOnInsert': {'created_at': now}, '$inc': {'revision': 1}},
                upsert=revision == 0, return_document=ReturnDocument.AFTER)
        except DuplicateKeyError:
            raise MemoryConflict('长期记忆已在其他位置更新，请重新加载后再保存。') from None
        if record is None:
            raise MemoryConflict('长期记忆已在其他位置更新，请重新加载后再保存。')
        return self._public(record)

    def patch(self, user_id, changes, *, revision):
        current = self.get(user_id)
        if current['revision'] != revision:
            raise MemoryConflict('长期记忆已更新，请重新读取后再操作。')
        facts = current['facts']
        for key, value in changes.items():
            if isinstance(value, list):
                facts[key] = list(dict.fromkeys([*facts[key], *value]))
            else:
                facts[key] = value
        return self.save(user_id, facts, revision=revision, source='explicit_chat')

    def forget(self, user_id, field='all', values=None, *, revision=None):
        current = self.get(user_id)
        if revision is not None and current['revision'] != revision:
            raise MemoryConflict('长期记忆已更新，请重新读取后再操作。')
        facts = current['facts']
        if field == 'all':
            facts = MemoryFacts().model_dump()
        elif field not in MemoryFacts.model_fields:
            raise MemoryDenied('Unknown memory field')
        elif isinstance(facts[field], list):
            facts[field] = [value for value in facts[field] if value not in values] if values else []
        else:
            facts[field] = ''
        # Keep a revision tombstone so stale form submissions cannot resurrect data.
        return self.save(user_id, facts, revision=current['revision'], source='explicit_delete')

    def ensure_thread(self, thread_id, user_id=None):
        _ = self.collection
        if user_id is None:
            owner = self._owners.find_one({'_id': thread_id})
            if owner:
                raise MemoryDenied('该会话属于其他用户，请创建新对话。')
            return
        self._validate_user(user_id)
        owner = self._owners.find_one_and_update({'_id': thread_id},
            {'$setOnInsert': {'user_id': user_id}}, upsert=True, return_document=ReturnDocument.AFTER)
        if owner['user_id'] != user_id:
            raise MemoryDenied('该会话属于其他用户，请创建新对话。')


def format_memory(profile):
    facts = {key: value for key, value in profile.get('facts', {}).items() if value}
    return json.dumps(facts, ensure_ascii=False)


def require_explicit_memory(text, changes):
    """Tools may only persist first-person, explicit requests grounded in this turn."""
    if (not re.search(r'记住|记下来|长期保存|长期记忆|以后都|今后都', text)
            or not re.search(r'我|本人|我们', text)
            or re.search(r'不要记住|别记住|不保存|不要保存|假设|举例|引用|他说|她说|这段话', text)):
        raise MemoryDenied('仅在用户明确要求长期记住本人偏好时保存；也可以在长期记忆面板中填写。')
    evidence = {'allergies': r'过敏|allerg', 'avoids': r'不吃|不要|忌|避免|不喜欢|不爱吃|avoid',
                'likes': r'喜欢|爱吃|偏好|like', 'diet': r'吃素|素食|清真|饮食方式|vegan|vegetarian',
                'goals': r'目标|减脂|增肌|控糖|高蛋白|体重|goal', 'budget': r'预算|元|块|budget'}
    for field, value in changes.items():
        if field not in evidence or not re.search(evidence[field], text, re.I):
            raise MemoryDenied('记忆字段必须有当前用户原话支持，不能推断。')
        for item in value if isinstance(value, list) else [value]:
            if not item or item.casefold() not in text.casefold():
                raise MemoryDenied('记忆内容必须摘自当前用户原话，不要扩展或改写。')
            clauses = [clause for clause in re.split(r'[，,。；;\n]', text) if item in clause]
            if field == 'likes' and any(re.search(r'不喜欢|不爱吃|讨厌', clause) for clause in clauses):
                raise MemoryDenied('不能将否定偏好保存为喜欢。')
            if field == 'allergies' and any(re.search(r'不过敏|没有.{0,8}过敏|不再.{0,8}过敏', clause) for clause in clauses):
                raise MemoryDenied('不能将否定过敏陈述保存为过敏。')


def require_explicit_forget(text, field, values):
    if (not re.search(r'忘记|清空|清除|删除|别再记|不要再记', text)
            or re.search(r'假设|举例|引用|他说|她说|这段话', text)):
        raise MemoryDenied('删除记忆需要用户明确提出。')
    if field == 'all':
        if not re.search(r'(?:清空|清除|删除)(?:我的)?(?:全部|所有)?(?:长期记忆|全部记忆|所有记忆)|忘记(?:我的)?(?:所有|全部)', text):
            raise MemoryDenied('请明确要清空全部长期记忆。')
    elif values:
        if any(value not in text for value in values):
            raise MemoryDenied('只能删除当前用户明确提到的内容。')
    else:
        labels = {'allergies': '过敏', 'avoids': '忌口', 'likes': '喜欢', 'diet': '饮食', 'goals': '目标', 'budget': '预算'}
        if labels.get(field, '\0') not in text:
            raise MemoryDenied('请明确要删除的记忆字段。')


user_memory_store = UserMemoryStore()
