"""Validated, versioned, progressively disclosed agent skills."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .errors import SkillError as HarnessSkillError


class SkillError(HarnessSkillError, ValueError):
    """Raised for an invalid skill definition or registry operation."""


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    name: str
    description: str
    version: str | None = None
    tags: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Skill:
    metadata: SkillMetadata
    instructions: str
    references: Mapping[str, Path] = field(default_factory=dict)
    resources: Mapping[str, Path] = field(default_factory=dict)
    script_files: Mapping[str, Path] = field(default_factory=dict)
    path: Path | None = None

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def identifier(self) -> str:
        return f"{self.name}@{self.metadata.version}" if self.metadata.version else self.name

    @property
    def scripts(self) -> tuple[str, ...]:
        return tuple(self.script_files)

    def context(self) -> str:
        """Return instructions only; references and resources remain on demand."""
        sections = [f"# Skill: {self.identifier}", self.instructions.strip()]
        if self.references:
            sections.append("Available references: " + ", ".join(self.references))
        if self.resources:
            sections.append("Available resources: " + ", ".join(self.resources))
        if self.script_files:
            sections.append("Available scripts: " + ", ".join(self.script_files))
        return "\n\n".join(filter(None, sections))

    def read_reference(self, name: str) -> str:
        path = self._asset(self.references, name, "reference")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SkillError(
                f"Skill '{self.identifier}' reference '{name}' must be UTF-8 text"
            ) from exc

    def read_resource(self, name: str) -> dict[str, Any]:
        path = self._asset(self.resources, name, "resource")
        data = path.read_bytes()
        try:
            return {"name": name, "encoding": "utf-8", "content": data.decode("utf-8")}
        except UnicodeDecodeError:
            return {
                "name": name,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
            }

    def script_path(self, name: str) -> Path:
        return self._asset(self.script_files, name, "script")

    def _asset(self, assets: Mapping[str, Path], name: str, kind: str) -> Path:
        try:
            return assets[name]
        except KeyError as exc:
            raise SkillError(
                f"Skill '{self.identifier}' has no {kind} named '{name}'"
            ) from exc


class SkillLoader:
    """Load and validate one conventional skill directory without eager assets."""

    def load(self, directory: str | Path) -> Skill:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise SkillError(f"Skill directory does not exist or is not a directory: {root}")
        skill_file = root / "SKILL.md"
        if not skill_file.is_file():
            raise SkillError(f"Skill '{root.name}' is missing SKILL.md: {root}")
        try:
            raw = skill_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SkillError(f"Skill '{root.name}' SKILL.md must be UTF-8: {skill_file}") from exc
        metadata_dict, instructions = self._parse_frontmatter(raw, skill_file)
        metadata = self._metadata(metadata_dict, skill_file)
        if not instructions.strip():
            raise SkillError(f"Skill '{metadata.name}' has empty instructions: {skill_file}")
        references = self._files(root, "references", metadata.name)
        resources = self._files(root, "resources", metadata.name)
        available_scripts = self._files(root, "scripts", metadata.name)
        declared_scripts: dict[str, Path] = {}
        for name in metadata.scripts:
            if name not in available_scripts:
                raise SkillError(
                    f"Skill '{metadata.name}' declares missing script '{name}' in {root / 'scripts'}"
                )
            self._validate_script(metadata.name, name)
            declared_scripts[name] = available_scripts[name]
        skill = Skill(
            metadata,
            instructions.strip(),
            MappingProxyType(references),
            MappingProxyType(resources),
            MappingProxyType(declared_scripts),
            root,
        )
        SkillValidator.validate_definition(skill)
        return skill

    @staticmethod
    def _files(root: Path, directory: str, skill_name: str) -> dict[str, Path]:
        asset_root = root / directory
        if not asset_root.exists():
            return {}
        if not asset_root.is_dir():
            raise SkillError(
                f"Skill '{skill_name}' {directory}/ path is not a directory: {asset_root}"
            )
        files: dict[str, Path] = {}
        for path in sorted(asset_root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(asset_root.resolve())
            except ValueError as exc:
                raise SkillError(
                    f"Skill '{skill_name}' {directory}/ contains an unsafe path: {path}"
                ) from exc
            files[path.relative_to(asset_root).as_posix()] = resolved
        return files

    @staticmethod
    def _validate_script(skill_name: str, name: str) -> None:
        suffix = Path(name).suffix.lower()
        if suffix not in {".py", ".ps1", ".sh"}:
            raise SkillError(
                f"Skill '{skill_name}' script '{name}' has unsupported extension; "
                "supported: .py, .ps1, .sh"
            )

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
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise SkillError(f"Skill metadata '{key}' must be a list of non-empty strings: {path}")
            values = tuple(item.strip() for item in value)
            if len(values) != len(set(values)):
                raise SkillError(f"Skill metadata '{key}' contains duplicates: {path}")
            return values

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
            scripts=strings("scripts"),
        )


class SkillValidator:
    """Apply the same explicit validation to loaded and programmatic skills."""

    @staticmethod
    def validate_definition(skill: Skill) -> None:
        identifier = skill.identifier
        if not skill.name.strip() or "@" in skill.name:
            raise SkillError(
                f"Skill '{identifier}' name must be non-empty and cannot contain '@'"
            )
        if not skill.metadata.description.strip():
            raise SkillError(f"Skill '{identifier}' description must be non-empty")
        if skill.metadata.version is not None and not skill.metadata.version.strip():
            raise SkillError(f"Skill '{skill.name}' version must not be empty")
        if not skill.instructions.strip():
            raise SkillError(f"Skill '{identifier}' instructions must be non-empty")
        for field_name, values in (
            ("tags", skill.metadata.tags),
            ("required_tools", skill.metadata.required_tools),
            ("dependencies", skill.metadata.dependencies),
            ("scripts", skill.metadata.scripts),
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise SkillError(
                    f"Skill '{identifier}' metadata '{field_name}' contains an invalid value"
                )
            if len(values) != len(set(values)):
                raise SkillError(
                    f"Skill '{identifier}' metadata '{field_name}' contains duplicates"
                )
        if set(skill.metadata.scripts) != set(skill.script_files):
            raise SkillError(
                f"Skill '{identifier}' declared scripts do not match validated scripts/ files"
            )
        SkillValidator._validate_assets(skill, "references", skill.references)
        SkillValidator._validate_assets(skill, "resources", skill.resources)
        SkillValidator._validate_assets(skill, "scripts", skill.script_files)

    @staticmethod
    def _validate_assets(skill: Skill, directory: str, assets: Mapping[str, Path]) -> None:
        if not assets:
            return
        if skill.path is None:
            raise SkillError(
                f"Skill '{skill.identifier}' has {directory} but no trusted skill directory"
            )
        root = (skill.path / directory).resolve()
        for name, path in assets.items():
            normalized = Path(name).as_posix()
            if Path(name).is_absolute() or normalized.startswith("../") or normalized == "..":
                raise SkillError(
                    f"Skill '{skill.identifier}' has unsafe {directory} name '{name}'"
                )
            resolved = Path(path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise SkillError(
                    f"Skill '{skill.identifier}' {directory} asset escapes its directory: {name}"
                ) from exc
            if not resolved.is_file():
                raise SkillError(
                    f"Skill '{skill.identifier}' {directory} asset is missing: {name}"
                )


class SkillRegistry:
    """One agent's isolated collection of versioned skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._by_name: dict[str, list[Skill]] = {}

    def register(self, skill: Skill) -> None:
        SkillValidator.validate_definition(skill)
        if skill.identifier in self._skills:
            raise SkillError(f"Duplicate skill: {skill.identifier}")
        self._skills[skill.identifier] = skill
        self._by_name.setdefault(skill.name, []).append(skill)

    def get(self, reference: str) -> Skill:
        if reference in self._skills:
            return self._skills[reference]
        candidates = self._by_name.get(reference, [])
        if not candidates:
            raise SkillError(f"Unknown skill: {reference}")
        if len(candidates) == 1:
            return candidates[0]
        return max(candidates, key=lambda item: self._version_key(item.metadata.version))

    @staticmethod
    def _version_key(version: str | None) -> tuple[Any, ...]:
        if version is None:
            return (0,)
        parts = re.split(r"([0-9]+)", version)
        return tuple((1, int(part)) if part.isdigit() else (0, part.lower()) for part in parts)

    def list(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())

    def validate(self) -> None:
        visiting: list[str] = []
        visited: set[str] = set()

        def visit(skill: Skill) -> None:
            identifier = skill.identifier
            if identifier in visiting:
                cycle = visiting[visiting.index(identifier) :] + [identifier]
                raise SkillError("Circular skill dependency: " + " -> ".join(cycle))
            if identifier in visited:
                return
            visiting.append(identifier)
            for dependency in skill.metadata.dependencies:
                try:
                    target = self.get(dependency)
                except SkillError as exc:
                    raise SkillError(
                        f"Skill '{identifier}' has missing dependency '{dependency}'"
                    ) from exc
                visit(target)
            visiting.pop()
            visited.add(identifier)

        for item in self._skills.values():
            visit(item)

    def load_order(self, reference: str, tool_names: set[str]) -> tuple[Skill, ...]:
        ordered: list[Skill] = []
        visited: set[str] = set()

        def visit(skill: Skill) -> None:
            if skill.identifier in visited:
                return
            for dependency in skill.metadata.dependencies:
                visit(self.get(dependency))
            missing = sorted(set(skill.metadata.required_tools) - tool_names)
            if missing:
                raise SkillError(
                    f"Skill '{skill.identifier}' requires unavailable tools: {', '.join(missing)}"
                )
            visited.add(skill.identifier)
            ordered.append(skill)

        visit(self.get(reference))
        return tuple(ordered)

    def dependency_identifiers(self, reference: str) -> tuple[str, ...]:
        """Return transitive dependencies in dependency-first order."""
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(skill: Skill) -> None:
            for dependency in skill.metadata.dependencies:
                target = self.get(dependency)
                if target.identifier in visited:
                    continue
                visit(target)
                visited.add(target.identifier)
                ordered.append(target.identifier)

        visit(self.get(reference))
        return tuple(ordered)

    def dependents_of(self, reference: str, candidates: list[str]) -> set[str]:
        identifier = self.get(reference).identifier
        return {
            candidate
            for candidate in candidates
            if identifier in self.dependency_identifiers(candidate)
        }

    def summaries(
        self, *, query: str | None = None, limit: int | None = None
    ) -> tuple[dict[str, str], ...]:
        skills = list(self._skills.values())
        if query:
            lowered = query.casefold()
            tokens = set(re.findall(r"[a-z0-9_-]+", lowered))
            query_cjk = self._cjk_ngrams(lowered)

            def score(skill: Skill) -> tuple[int, str]:
                fields = " ".join(
                    [skill.name, skill.metadata.description, *skill.metadata.tags]
                ).casefold()
                field_tokens = set(re.findall(r"[a-z0-9_-]+", fields))
                field_cjk = self._cjk_ngrams(fields)
                points = len(tokens.intersection(field_tokens)) * 4
                points += sum(
                    3 if len(gram) == 3 else 2
                    for gram in query_cjk.intersection(field_cjk)
                )
                points += sum(
                    6
                    for value in (skill.name, *skill.metadata.tags)
                    if value.casefold() in lowered
                )
                points += 8 if lowered in fields else 0
                points += 4 if skill.metadata.description.casefold() in lowered else 0
                return points, skill.identifier

            ranked = sorted(skills, key=score, reverse=True)
            matched = [item for item in ranked if score(item)[0] > 0]
            skills = matched or ranked
        if limit is not None:
            skills = skills[:limit]
        return tuple(
            {"name": skill.identifier, "description": skill.metadata.description}
            for skill in skills
        )

    @staticmethod
    def _cjk_ngrams(text: str) -> set[str]:
        grams: set[str] = set()
        for sequence in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text):
            for size in (2, 3):
                grams.update(
                    sequence[index : index + size]
                    for index in range(len(sequence) - size + 1)
                )
        return grams


@dataclass(frozen=True, slots=True)
class ScriptResult:
    skill: str
    script: str
    status: str
    result: Any = None
    error: str | None = None
    stderr: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "script": self.script,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "stderr": self.stderr,
        }


class SkillScriptRunner:
    """Execute declared scripts only, using JSON stdin/stdout as one entry contract."""

    def __init__(self, timeout_seconds: float = 30.0, max_output_bytes: int = 1_000_000) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def run(self, skill: Skill, script: str, arguments: Mapping[str, Any] | None = None) -> ScriptResult:
        path = skill.script_path(script).resolve()
        if skill.path is None:
            raise SkillError(f"Skill '{skill.identifier}' has no trusted directory")
        scripts_root = (skill.path / "scripts").resolve()
        try:
            path.relative_to(scripts_root)
        except ValueError as exc:
            raise SkillError(
                f"Skill '{skill.identifier}' script escapes scripts directory: {script}"
            ) from exc
        command = self._command(path)
        try:
            payload = json.dumps(dict(arguments or {}), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise SkillError(
                f"Skill '{skill.identifier}' script arguments are not JSON serializable: {exc}"
            ) from exc
        try:
            completed = subprocess.run(
                command,
                input=payload,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=str(scripts_root),
                env=dict(os.environ),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ScriptResult(
                skill.identifier,
                script,
                "error",
                error=f"timed out after {self.timeout_seconds:g}s",
            )
        stdout = completed.stdout
        stderr = completed.stderr.strip() or None
        if len(stdout.encode("utf-8")) > self.max_output_bytes:
            return ScriptResult(
                skill.identifier,
                script,
                "error",
                error="script output exceeded size limit",
                stderr=stderr,
            )
        if completed.returncode != 0:
            return ScriptResult(
                skill.identifier,
                script,
                "error",
                error=f"script exited with code {completed.returncode}",
                stderr=stderr,
            )
        output = stdout.strip()
        if not output:
            result: Any = None
        else:
            try:
                result = json.loads(output)
            except json.JSONDecodeError:
                result = output
        return ScriptResult(skill.identifier, script, "success", result=result, stderr=stderr)

    @staticmethod
    def _command(path: Path) -> list[str]:
        suffix = path.suffix.lower()
        if suffix == ".py":
            return [sys.executable, str(path)]
        if suffix == ".ps1":
            executable = shutil.which("pwsh") or shutil.which("powershell")
            if not executable:
                raise SkillError("PowerShell is unavailable for .ps1 skill script")
            return [executable, "-NoProfile", "-NonInteractive", "-File", str(path)]
        if suffix == ".sh":
            executable = shutil.which("bash")
            if not executable:
                raise SkillError("bash is unavailable for .sh skill script")
            return [executable, str(path)]
        raise SkillError(f"Unsupported skill script extension: {path.suffix}")
