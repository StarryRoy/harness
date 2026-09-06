"""Skill models, on-disk loading, and per-agent registration."""

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


class SkillError(ValueError):
    """Raised for an invalid skill definition or registry operation."""


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    name: str
    description: str
    version: str | None = None
    tags: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Skill:
    metadata: SkillMetadata
    instructions: str
    references: Mapping[str, str] = field(default_factory=dict)
    path: Path | None = None

    @property
    def name(self) -> str:
        return self.metadata.name

    def context(self) -> str:
        sections = [f"# Skill: {self.name}", self.instructions.strip()]
        for name, content in self.references.items():
            sections.append(f"## Reference: {name}\n{content.strip()}")
        return "\n\n".join(sections)


class SkillLoader:
    """Load a SKILL.md and text references from a conventional directory."""

    def load(self, directory: str | Path) -> Skill:
        root = Path(directory).expanduser().resolve()
        skill_file = root / "SKILL.md"
        if not skill_file.is_file():
            raise SkillError(f"Missing SKILL.md in {root}")
        raw = skill_file.read_text(encoding="utf-8")
        metadata_dict, instructions = self._parse_frontmatter(raw, skill_file)
        metadata = self._metadata(metadata_dict, skill_file)
        references: dict[str, str] = {}
        reference_dir = root / "references"
        if reference_dir.exists():
            if not reference_dir.is_dir():
                raise SkillError(f"references is not a directory: {reference_dir}")
            for path in sorted(reference_dir.rglob("*")):
                if path.is_file():
                    try:
                        references[path.relative_to(reference_dir).as_posix()] = path.read_text(
                            encoding="utf-8"
                        )
                    except UnicodeDecodeError as exc:
                        raise SkillError(f"Reference must be UTF-8 text: {path}") from exc
        return Skill(metadata, instructions.strip(), MappingProxyType(references), root)

    @staticmethod
    def _parse_frontmatter(raw: str, path: Path) -> tuple[dict[str, Any], str]:
        lines = raw.splitlines()
        if not lines or lines[0].strip() != "---":
            raise SkillError(f"SKILL.md must start with YAML frontmatter: {path}")
        try:
            end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
        except StopIteration as exc:
            raise SkillError(f"Unclosed YAML frontmatter: {path}") from exc
        try:
            parsed = yaml.safe_load("\n".join(lines[1:end])) or {}
        except yaml.YAMLError as exc:
            raise SkillError(f"Invalid YAML metadata in {path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SkillError(f"Skill metadata must be a mapping: {path}")
        return parsed, "\n".join(lines[end + 1 :])

    @staticmethod
    def _metadata(data: dict[str, Any], path: Path) -> SkillMetadata:
        for key in ("name", "description"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise SkillError(f"Skill metadata '{key}' must be a non-empty string: {path}")

        def strings(key: str) -> tuple[str, ...]:
            value = data.get(key, [])
            if value is None:
                return ()
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise SkillError(f"Skill metadata '{key}' must be a list of strings: {path}")
            return tuple(value)

        version = data.get("version")
        if version is not None and not isinstance(version, (str, int, float)):
            raise SkillError(f"Skill metadata 'version' must be a scalar: {path}")
        return SkillMetadata(
            name=data["name"].strip(),
            description=data["description"].strip(),
            version=str(version) if version is not None else None,
            tags=strings("tags"),
            required_tools=strings("required_tools"),
            dependencies=strings("dependencies"),
        )


class SkillRegistry:
    """Agent-scoped collection of uniquely named skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise SkillError(f"Duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillError(f"Unknown skill: {name}") from exc

    def list(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())

    def summaries(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {"name": skill.name, "description": skill.metadata.description}
            for skill in self._skills.values()
        )
