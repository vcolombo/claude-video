# Watch skill

Agent Skills package that gives an agent video input (download → frames → transcript) via the `/watch` slash command.

## Language

**Plugin**:
The installable unit: manifests plus the skills it ships. This repo is a plugin.
_Avoid_: bundle, package

**Skill**:
One model-invoked capability: a `SKILL.md` contract plus its `scripts/`. This repo ships one skill, `watch`.
_Avoid_: command, wrapper

**Install surface**:
One host smell: Claude Code, Codex, omp, claude.ai. Each surface gets its own manifest pointing at the same self-contained `skills/watch/` folder.
_Avoid_: platform, target
