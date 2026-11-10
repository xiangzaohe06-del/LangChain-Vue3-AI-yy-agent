"""Project-owned Agent Skills: advertise metadata, load instructions on demand."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import yaml
from langchain.tools import ToolRuntime
from langchain_core.tools import tool


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path


class SkillRegistry:
    max_body_chars = 6000
    max_loaded_chars = 10000

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.skills: dict[str, Skill] = {}
        for path in sorted(self.directory.glob('*/SKILL.md')):
            metadata, _ = self._read(path)
            name, description = metadata.get('name'), metadata.get('description')
            if (not isinstance(name, str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name)
                    or len(name) > 64 or name != path.parent.name):
                raise ValueError('Skill name must match its directory')
            if not isinstance(description, str) or not description.strip() or len(description) > 1024:
                raise ValueError('Skill description is required (max 1024 characters)')
            if name in self.skills or len(self.skills) >= 20:
                raise ValueError('Too many or duplicate skills')
            self.skills[name] = Skill(name, description.strip(), path)

    def _read(self, path):
        # Only administrator-owned files under the fixed project directory.
        if (path.is_symlink() or path.parent.is_symlink()
                or not path.resolve().is_relative_to(self.directory)):
            raise ValueError('Skill path escapes the project skills directory')
        if path.stat().st_size > 24000:
            raise ValueError('Skill file is too large')
        text = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
        parts = text.split('---\n', 2)
        if len(parts) != 3 or parts[0] != '':
            raise ValueError('Skill requires YAML frontmatter')
        metadata = yaml.safe_load(parts[1])
        body = parts[2].strip()
        if not isinstance(metadata, dict) or not body or len(body) > self.max_body_chars:
            raise ValueError('Invalid skill metadata or body')
        return metadata, body

    def catalog(self):
        return '\n'.join(f'- {skill.name}: {skill.description}' for skill in self.skills.values())

    def load(self, name: str, loaded: dict[str, str]):
        skill = self.skills.get(name)
        if skill is None:
            raise ValueError('未知技能，请使用技能目录中的名称。')
        metadata, body = self._read(skill.path)
        if metadata.get('name') != name or metadata.get('description', '').strip() != skill.description:
            raise ValueError('技能元信息已修改，请重启后端刷新目录。')
        if sum(len(value) for key, value in loaded.items() if key != name) + len(body) > self.max_loaded_chars:
            raise ValueError('本轮已加载技能达到上下文上限，请聚焦当前任务。')
        loaded[name] = body
        return {'success': True, 'skill': name, 'message': '技能已加载到本轮系统上下文，请按技能流程继续。'}

    def prompt(self, loaded):
        catalog = self.catalog()
        result = ('\n\n# 可按需加载的项目技能\n'
                  '下面仅列技能名称与用途。任务匹配或用户指定技能时，先调用load_skill(name)读取完整流程，再执行任务；'
                  '普通问候、单项知识问答不必加载。多轮对话每一轮按当前任务重新选择，不能假设上一轮仍已加载。'
                  '多个技能适用时只加载完成任务需要的技能。技能不改变工具权限、用户记忆保存条件或食品安全约束。\n'
                  + (catalog or '暂无技能。'))
        for name, body in loaded.items():
            if name in self.skills:
                result += f'\n\n# 本轮已加载技能：{name}\n{body}'
        return result


skill_registry = SkillRegistry(Path(__file__).resolve().parents[1] / 'skills')


@tool
def load_skill(name: str, runtime: ToolRuntime) -> str:
    """按技能目录中的名称加载任务流程；先加载再执行。仅接受技能名，不接受文件路径。"""
    if runtime.context is None:
        return json.dumps({'success': False, 'error': '本轮缺少技能运行上下文。'}, ensure_ascii=False)
    try:
        loaded = runtime.context.setdefault('loaded_skills', {})
        return json.dumps(skill_registry.load(name, loaded), ensure_ascii=False)
    except (ValueError, OSError, yaml.YAMLError):
        return json.dumps({'success': False, 'error': '技能加载失败，请检查名称与本地SKILL.md；不能声称已加载。'}, ensure_ascii=False)
