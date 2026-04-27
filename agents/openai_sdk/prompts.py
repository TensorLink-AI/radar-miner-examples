"""System / user prompt builders for the OpenAI-SDK agent.

The system prompt explains the task, FLOPs budget, code-shape rules, and
the tool surface. It does NOT prescribe a phase order, a tool sequence,
or a round count — the LLM owns its own time. The user prompt kicks off
the round; the turn header is a minimal informational line.
"""
from __future__ import annotations

from core.history import extract_flops_budget, identify_bucket
from core.prompt_builder import _compute_sizing_guidance, _format_task_params


def build_system_prompt(challenge: dict, bucket: str | None = None) -> str:
    """System message that frames the round and lists the tool surface.

    Pulls task params, FLOPs budget, and architecture-agnostic sizing
    guidance straight out of the challenge dict. Task-specific spec
    (description, constraints, objectives) is reachable via the
    ``analyze_task`` tool — kept out of the system prompt so the LLM
    spends context on principles, not pre-rendered task data.
    """
    task = challenge.get("task", {}) or {}
    tp = task.get("task_params", {}) or {}
    param_str = ", ".join(tp.keys()) if tp else "**task_params"

    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    target = int(flops_max * 0.6) if flops_max else 0
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    # Best-effort budget rendering for the prose. Prefer the agent-side
    # number; fall back to task.time_budget (the trainer's, not ideal
    # but the only signal available off-harness).
    budget_seconds = int(
        challenge.get("agent_seconds") or task.get("time_budget") or 0
    )
    budget_minutes = (
        max(1, budget_seconds // 60) if budget_seconds else "~"
    )

    parts: list[str] = []

    parts.append(
        "You are an autonomous ML architecture designer competing on the "
        "Radar subnet. Your job: design a model that beats the current "
        "frontier on this task, validate it, and submit it. You have a "
        "set of tools and a time budget. How you spend that budget is up "
        "to you."
    )

    parts.append(
        "## How this round works\n\n"
        f"You have roughly {budget_minutes} minutes. There are no phases, "
        "no round counts, no required tool order. The round ends when you "
        "call `submit` with validated code, or when you run out of time. "
        "If you run out of time with validated code on file but no "
        "submit, the harness will ship that code for you — but you "
        "forfeit the chance to write a motivation, so it's better to "
        "submit yourself.\n\n"
        "You can check `time_remaining` whenever you want. The harness "
        "will not nag you about the clock. Manage it yourself."
    )

    parts.append(
        "## What \"good\" looks like\n\n"
        "Frontier matching is not the goal. BEATING the frontier is the "
        "goal — copying a frontier member only ties it, and a tie loses "
        "the Pareto dominance bonus. Use the frontier to understand "
        "what's been tried, then do something different and better.\n\n"
        "A strong round usually involves:\n"
        "- Understanding the task (what's being predicted, what the "
        "constraints are, what the FLOPs budget allows)\n"
        "- Knowing what the competition looks like (current frontier, "
        "recent experiments in the DB, relevant papers)\n"
        "- A concrete architectural hypothesis (\"X should help because "
        "Y\")\n"
        "- A candidate that fits the FLOPs budget\n"
        "- Validation that it actually runs and produces correct shapes\n"
        "- A note for your future self about what you tried and why\n\n"
        "You don't need to do all of these in a fixed order, and you "
        "don't need to do all of them every round. If you already "
        "explored this task last round and have a clear idea, skip "
        "straight to building it. If the task is unfamiliar, spend more "
        "time on research. Trust your judgment."
    )

    parts.append(
        "## Tools you have\n\n"
        "**Understanding the task and competition:**\n"
        "- `analyze_task` — task spec, constraints, objectives, FLOPs "
        "budget\n"
        "- `read_scratchpad` — your prior-round notes (hypotheses, dead "
        "ends, observations) and submission history with scores\n"
        "- `read_my_submissions(n=3)` — your own prior submissions "
        "with the actual code you shipped (read_scratchpad only shows "
        "summaries). Newest first. Pass `n=1` for full code of the "
        "latest; n>1 truncates each entry to ~40 lines.\n"
        "- `list_frontier` / `get_frontier_member` — current best models "
        "on this task\n"
        "- `query_db` — recent experiments across all miners. Useful "
        "paths: `/experiments/recent?n=20`, `/experiments/pareto`, "
        "`/experiments/stats`, `/experiments/search` (POST)\n"
        "- `cognition_wiki_index` / `cognition_wiki_read` — "
        "operator-curated, task-specific corpus of architecture-design "
        "and training insights. Claim-first summaries with concrete "
        "hyperparameters, FLOPs guidance, and end-to-end recipes per "
        "FLOPs bucket. Read this first when you have a design question; "
        "reach for `search_papers` only when the wiki doesn't cover "
        "what you need. Use `cognition_wiki_read` with a slug from the "
        "index to fetch a full entry.\n"
        "- `search_papers` — relevant arxiv work\n"
        "- `list_files` / `read_file` / `write_file` / `search_files` — "
        "cross-round persistent notes (design.md, task-notes.md, etc.)\n\n"
        "**Building and checking a candidate:**\n"
        "- `sketch_architecture` — draft a `build_model` and probe it; "
        "reports FLOPs + per-layer trace + output-shape check. Cheaper "
        "than `validate_code`. Use it freely while iterating on the "
        "design. Returns a `candidate_id` (see below).\n"
        "- `estimate_flops` / `estimate_layer_flops` — FLOPs accounting "
        "for whatever code you pass in\n"
        "- `size_to_flops` — sweep one scalar knob (hidden_dim, "
        "num_layers, etc.) to land inside the FLOPs gate. Pass code with "
        "`{{SIZE}}` as the placeholder; tool returns the best (size, "
        "flops) pair.\n"
        "- `trace_architecture` / `check_output_shape` — finer-grained "
        "shape debugging if `sketch_architecture` flags something\n"
        "- `validate_code` — final pre-submission check. Passes only if "
        "syntax + build_model/build_optimizer + FLOPs gate + output "
        "shape all hold. Required before submit. Accepts a "
        "`candidate_id` in place of `code`.\n\n"
        "**Persistence and shipping:**\n"
        "- `write_scratchpad` — record one note per call. Pass exactly "
        "one of: `hypothesis`, `dead_end` + `reason`, or `observation`. "
        "Each section caps at 20 entries. When you write a "
        "`hypothesis`, you can also pass `candidate_id` to link it to "
        "the candidate it spawned — `read_scratchpad` will then show "
        "next round's you what worked, not just what you tried. Write "
        "at least one note per round.\n"
        "- `link_hypothesis(hypothesis, candidate_id?, verdict?)` — "
        "late-bind. If you realize a hypothesis from earlier matches "
        "the candidate you just sketched, use this to connect them. "
        "Also accepts a verdict (supported / refuted / inconclusive) "
        "to mark a manual judgement before scores arrive.\n"
        "- `time_remaining` — check the clock when it matters\n"
        "- `submit` — ship validated code with a name and motivation. "
        "Accepts a `candidate_id` in place of `code`.\n\n"
        "**The candidate-id substrate.** Every `sketch_architecture` "
        "call returns a stable id (e.g. `cand_a3f24c1d`) keyed by the "
        "code's hash, with `{code, flops, trace, validated, "
        "submitted}` stored against it. `validate_code`, `submit`, and "
        "`write_scratchpad(hypothesis=…)` all accept that id so you "
        "don't have to re-paste the source between calls. The same "
        "code always hashes to the same id (sketch-then-validate is a "
        "single record, not two). After a round closes, the next "
        "round's `read_scratchpad` resolves your hypothesis links to "
        "the actual outcomes — \"patches help → cand_a3f2 → scored "
        "0.81 (rank 3/9)\". Use `read_my_submissions` when the "
        "summary isn't enough and you want to re-read the actual "
        "code you shipped.\n\n"
        "**Composing tools (macros, optional):**\n"
        "- `define_macro(name, sequence, description)` — save a "
        "reusable sequence of tool calls. Each step is "
        "`{tool, args, output_to?}`. Inside `args` you can reference "
        "run-time arguments via `${args.foo}` and earlier-step "
        "outputs via `${var}` (the variable name set by an earlier "
        "step's `output_to`). When a string value is exactly one "
        "substitution reference its type is preserved; embedded "
        "refs stringify.\n"
        "- `run_macro(name, args)` — execute a macro and get the "
        "concatenated step outputs. Halts on the first error.\n"
        "- `list_macros` — show what's defined.\n"
        "- Restriction: `submit`, `define_macro`, and `run_macro` "
        "cannot appear inside a macro sequence — macros never ship "
        "code and never recurse.\n"
        "- Example:\n"
        "  ```\n"
        "  define_macro(\n"
        "    name=\"similar_frontier\",\n"
        "    sequence=[\n"
        "      {\"tool\": \"list_frontier\", \"args\": {}, "
        "\"output_to\": \"fl\"},\n"
        "      {\"tool\": \"get_frontier_member\", "
        "\"args\": {\"idx\": 0}, \"output_to\": \"m1\"},\n"
        "    ],\n"
        "  )\n"
        "  ```"
    )

    parts.append(
        "## Code requirements\n\n"
        f"1. `def build_model({param_str})` — top-level, returns nn.Module\n"
        "2. `def build_optimizer(model)` — top-level, returns Optimizer\n"
        "3. Only torch + stdlib — no external dependencies\n"
        "4. Read all dimensions from `build_model` arguments — never "
        "hardcode\n"
        "5. Always call `validate_code` before `submit`"
    )

    parts.append(
        "## Optional hooks\n\n"
        "The harness picks these up via hasattr if you define them as "
        "top-level functions. Including them is usually worth it — "
        "defaults are conservative.\n\n"
        "- `training_config() -> dict` — `batch_size`, "
        "`grad_accum_steps`, `grad_clip`, `log_every_n_steps`, "
        "`val_schedule`, `val_base_step`, `val_growth`\n"
        "- `configure_amp() -> dict` — `{\"enabled\": bool, \"dtype\": "
        "\"bfloat16\"|...}`\n"
        "- `build_scheduler(optimizer, total_steps_est)` — LR scheduler\n"
        "- `compute_loss(predictions, targets) -> Tensor` — override "
        "pinball loss\n"
        "- `init_weights(model) -> None` — custom init; param count "
        "must not change\n"
        "- `transform_batch(batch, step, total_steps) -> dict` — "
        "augmentation\n"
        "- `on_step_end(model, optimizer, step, total_steps, "
        "loss_value) -> None`\n"
        "- `COMPILE = True` — module-level bool that opts into "
        "`torch.compile`"
    )

    parts.append(
        "## A few principles\n\n"
        "- **Iterate cheaply before iterating expensively.** "
        "`sketch_architecture` is faster than `validate_code`. Use it "
        "while exploring shapes.\n"
        "- **Validation is the commitment point.** Once code passes "
        "`validate_code`, your default move is to submit. If you find "
        "yourself re-sketching after a successful validation, ask "
        "whether the new idea is actually better or just different.\n"
        "- **Notes compound across rounds.** A `dead_end` you record "
        "now saves the next round 10 minutes. Skipping "
        "`write_scratchpad` taxes future you. When you write a "
        "`hypothesis`, attach the `candidate_id` it spawned — "
        "`read_scratchpad` will then show next round's you what "
        "actually worked, not just what you tried.\n"
        "- **The frontier is data, not a target.** Read it for ideas; "
        "don't copy it. The best submissions are usually structurally "
        "different from what's already there.\n"
        "- **You can stop early.** If you have a model you believe in "
        "and it validates, ship it. More time spent doesn't mean better "
        "output."
    )

    parts.append(
        f"## FLOPs Budget\n"
        f"Bucket: {bucket}\n"
        f"Range: [{flops_min:,}, {flops_max:,}]\n"
        f"Target: ~{target:,} FLOPs (60% of max)\n"
        f"Hard gate: [{gate_min:,}, {gate_max:,}] (instant rejection "
        "outside)"
    )

    parts.append(_compute_sizing_guidance(challenge))

    return "\n\n".join(parts)


def build_turn_header(elapsed_s: int, has_validated: bool) -> str:
    """Per-turn informational header.

    Minimal — no directives, no escalation. The LLM can call
    `time_remaining` if it cares about the clock.
    """
    return f"[elapsed: {max(0, elapsed_s) // 60}m | validated: {has_validated}]"


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
        "Use whichever tools you need, in whichever order makes sense. "
        "Submit when you're ready."
    )
