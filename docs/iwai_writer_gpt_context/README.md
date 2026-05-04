# IWAI / Claude Handoff Context Pack

Use this folder to brief Claude or another web writing assistant on the current project state.

Recommended minimal upload:

1. `claude_handoff_context.txt`
2. `gpt_system_instructions.txt`
3. `iwai_style_guide.txt`
4. `codebase_map_for_claude.txt` if discussing experiments, metrics, or implementation feasibility
5. current LaTeX sections from `../../../../thesis-report/sections/`, especially `04_efe_planning.tex`, `05_gp_observability.tex`, and `06_experiments.tex`

Optional upload if there is room:

1. `paper_outline.txt`
2. `experiment_blueprint.txt`
3. `claims_and_caveats.txt`
4. `starter_prompts.txt`

Main intended use:

- brainstorm the final experiment protocol and metrics;
- critique whether proposed claims are supported;
- revise paper text in a compact IWAI/LNCS style;
- reason about what the current cleaned code actually supports;
- keep the method story honest: learned observability changes `R_eff`, not a direct visibility reward.

Do not upload old logs indiscriminately. If Claude needs numbers, provide a small run summary table or selected current plots, not the full historical mess.
