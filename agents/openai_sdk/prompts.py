"""System / user prompt builders for the OpenAI-SDK agent.

Wraps the existing ``core.prompt_builder`` helpers so the OpenAI agent
gets the same sizing/shape guidance the autonomous agent uses without
duplicating that logic. The system prompt explains the task, FLOPs
budget, code-shape rules, and the tool-calling contract; the user
prompt kicks off the round.
"""
from __future__ import annotations

from core.history import extract_flops_budget, identify_bucket
from core.prompt_builder import _compute_sizing_guidance, _format_task_params


def build_system_prompt(challenge: dict, bucket: str | None = None) -> str:
    """System message that teaches the LLM how to use the agent's tools.

    Pulls task params, constraints, FLOPs budget, and architecture-agnostic
    sizing guidance straight out of the challenge dict.
    """
    task = challenge.get("task", {}) or {}
    tp = task.get("task_params", {}) or {}
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    task_name = task.get("name", "unknown")

    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    target = int(flops_max * 0.6) if flops_max else 0
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    parts: list[str] = []

    parts.append(
        "You are an autonomous ML architecture designer competing on the "
        "Radar subnet. You have a small set of tools — research the task, "
        "sketch a candidate, size it to the FLOPs budget, validate, and "
        "submit. The loop ends when you call `submit` with validated "
        "code, or when the time budget runs out.\n\n"
        "Frontier matching is not the goal. BEATING the frontier is the "
        "goal — copying a frontier member only ties it, and a tie loses "
        "the Pareto dominance bonus. Use the frontier to understand "
        "what's been tried, not as a blueprint.\n\n"
        "Recommended workflow (each step feeds the next):\n"
        "  1. `analyze_task` — read the task spec, constraints, "
        "objectives, and FLOPs budget.\n"
        "  2. `read_scratchpad` — prior-round history (scored), "
        "open hypotheses, dead ends, task observations. Do this BEFORE "
        "drafting so you don't repeat a dead end.\n"
        "  3. `sketch_architecture` — draft a `build_model` and probe "
        "it. This is cheaper than `validate_code` and reports FLOPs + "
        "per-layer trace + output-shape check. Sketch FIRST, before "
        "peeking at the frontier, so your idea isn't anchored to an "
        "existing design.\n"
        "  4. Research the competition (2-3 turns max):\n"
        "     - `list_frontier` / `get_frontier_member` for current "
        "best models — what ops do they use, where is your sketch "
        "different, what is the weakest existing score?\n"
        "     - `cognition_wiki_index` for the operator-curated, "
        "task-specific corpus of architecture-design and training "
        "insights — claim-first summaries with concrete "
        "hyperparameters, FLOPs guidance, and end-to-end recipes "
        "per FLOPs bucket. Read this first when you have a design "
        "question; reach for `search_papers` only when the wiki "
        "doesn't cover what you need. Use `cognition_wiki_read` "
        "with a slug from the index to fetch a full entry.\n"
        "     - `query_db` for recent-experiment context the frontier "
        "doesn't show. Useful paths: `/experiments/recent?n=20` "
        "(what's been tried lately and how it scored), "
        "`/experiments/pareto` (Pareto-optimal experiments by task), "
        "`/experiments/stats` (DB-wide stats), "
        "`/experiments/search` (POST to search by query). A DB call "
        "is cheap — use it every round. Skipping it means you're "
        "flying blind on what miners around you are trying.\n"
        "     - `search_papers` for arxiv work relevant to the task.\n"
        "     Aim to dominate the frontier, not match it.\n"
        "  5. `size_to_flops` — once the shape of your design is "
        "settled, sweep a scalar knob (hidden_dim, num_layers, "
        "channels, ...) to land inside the FLOPs gate. Supply a code "
        "template with `{{SIZE}}`; the tool returns the best measured "
        "(size, flops) pair.\n"
        "  6. `validate_code` — final pre-submission check. Passes "
        "only if syntax + build_model/build_optimizer + FLOPs gate + "
        "output shape all hold.\n"
        "  7. `write_scratchpad` — record at least one note "
        "(`hypothesis`, `dead_end` + `reason`, or `observation`) so "
        "the next round learns from this one. REQUIRED before submit.\n"
        "  8. `submit` — ship validated code with a short name and a "
        "motivation.\n\n"
        "Persistent state: `read_scratchpad` returns your submission "
        "history (annotated with validator-reported scores when "
        "available) and three structured note sections — "
        "`open_hypotheses`, `dead_ends`, `task_observations`. Write to "
        "those sections via `write_scratchpad` with ONE of: "
        "`hypothesis`, `dead_end` + `reason`, or `observation`. Each "
        "section is capped at 20 entries. For longer structured notes "
        "(design.md, task-notes.md, etc.) use `list_files`, "
        "`read_file`, and `write_file` — these persist across rounds "
        "alongside the scratchpad."
    )

    parts.append(
        f"## Task: {task_name}\n"
        + (task.get("description", "") + "\n\n" if task.get("description") else "")
        + f"Task parameters: {_format_task_params(tp)}\n"
        f"build_model signature: `def build_model({param_str})`"
    )

    domain = task.get("domain_system_prompt", "")
    if domain:
        parts.append(f"## Domain Context\n{domain}")

    constraints = task.get("constraints", []) or []
    if constraints:
        parts.append(
            "## Constraints\n" + "\n".join(f"- {c}" for c in constraints)
        )

    objectives = task.get("objectives", []) or []
    if objectives:
        parts.append(
            "## Objectives\n" + "\n".join(f"- {o}" for o in objectives)
        )

    parts.append(
        f"## FLOPs Budget\n"
        f"Bucket: {bucket}\n"
        f"Range: [{flops_min:,}, {flops_max:,}]\n"
        f"Target: ~{target:,} FLOPs (60% of max)\n"
        f"Hard gate: [{gate_min:,}, {gate_max:,}] (instant rejection outside)"
    )

    parts.append(_compute_sizing_guidance(challenge))

    parts.append(
        "## Code Requirements\n"
        f"1. `def build_model({param_str})` — top-level, returns nn.Module\n"
        "2. `def build_optimizer(model)` — top-level, returns Optimizer\n"
        "3. Only torch + stdlib — no external dependencies\n"
        "4. Read all dimensions from the build_model arguments — never "
        "hardcode shapes\n"
        "5. Always call `validate_code` before `submit`"
    )

    parts.append(
        "## Optional Hooks (the harness picks these up via hasattr)\n"
        "Define any of these as top-level functions to control training "
        "dynamics. Including them is usually worth it — defaults are "
        "conservative.\n"
        "- `training_config() -> dict` — `batch_size`, `grad_accum_steps`, "
        "`grad_clip`, `log_every_n_steps`, `val_schedule`, `val_base_step`, "
        "`val_growth` (clamped to sane ranges by the harness).\n"
        "- `configure_amp() -> dict` — `{\"enabled\": bool, \"dtype\": "
        "\"bfloat16\"|\"float16\"|\"float32\"}`.\n"
        "- `build_scheduler(optimizer, total_steps_est)` — LR scheduler "
        "(warmup + cosine goes here).\n"
        "- `compute_loss(predictions, targets) -> Tensor` — override the "
        "default pinball loss.\n"
        "- `init_weights(model) -> None` — custom weight init; param count "
        "must NOT change.\n"
        "- `transform_batch(batch, step, total_steps) -> dict` — "
        "batch-level augmentation; `batch` has `\"input\"`/`\"target\"`.\n"
        "- `on_step_end(model, optimizer, step, total_steps, loss_value) "
        "-> None` — post-step hook (EMA, freezing, logging).\n"
        "- `COMPILE = True` — module-level bool that opts into "
        "`torch.compile`."
    )

    return "\n\n".join(parts)


def build_turn_header(
    remaining_s: int, phase: str, has_validated: bool = False,
) -> str:
    """Per-turn directive injected at the top of every LLM turn.

    The text changes based on remaining seconds and whether the agent
    already has validated code stashed. This is the primary lever
    against the "analysis paralysis" failure mode: as the deadline
    approaches the message escalates from "normal pace" to an explicit
    SUBMIT-NOW order so the LLM stops re-sketching and ships.
    """
    if has_validated:
        if remaining_s < 100:
            directive = (
                "EMERGENCY: call write_scratchpad (one note) then submit "
                "with the validated code RIGHT NOW."
            )
        elif remaining_s < 300:
            directive = (
                "You have validated code. Call write_scratchpad (one "
                "note) then submit IMMEDIATELY. Do not sketch or "
                "research again."
            )
        else:
            directive = (
                "You have validated code. Stop exploring. Next: "
                "write_scratchpad + submit."
            )
    else:
        if remaining_s < 100:
            directive = (
                "EMERGENCY: submit the best code you've produced this "
                "round, even if unvalidated. Empty-handed = lost round."
            )
        elif remaining_s < 300:
            directive = (
                "SUBMIT MODE: if you have any candidate, go straight to "
                "validate_code → submit. Skip further research and "
                "re-sketching."
            )
        elif remaining_s < 600:
            directive = (
                "Time is short. If you haven't sketched, sketch NOW. "
                "If you have, go straight to size_to_flops + validate_code."
            )
        else:
            directive = (
                "Normal pace. Sketch once, one research call, size, "
                "validate, submit."
            )
    return f"[REMAINING: {remaining_s}s | PHASE: {phase}] {directive}"


def build_user_prompt(challenge: dict, bucket: str | None = None) -> str:
    """Initial user message that starts the round."""
    task = challenge.get("task", {}) or {}
    task_name = task.get("name", "unknown")
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    frontier = (
        challenge.get("feasible_frontier")
        or challenge.get("pareto_frontier")
        or []
    )

    return (
        f"New round started. Task: {task_name}, bucket: {bucket}, "
        f"FLOPs range: [{flops_min:,}, {flops_max:,}]. "
        f"Frontier has {len(frontier)} model(s) to beat (beat, not "
        "match — a tie forfeits the dominance bonus).\n\n"
        "Start by reading the task and your scratchpad. Sketch before "
        "you peek at the frontier, then research the competition — "
        "`list_frontier`, `query_db` (recent experiments / Pareto / "
        "stats), and `search_papers` — so you know what's been tried. "
        "Then size the sketch to the FLOPs budget, validate, note what "
        "you learned, and submit."
    )
