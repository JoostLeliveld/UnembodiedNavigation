# Custom GPT Context Pack

These files are intended to be uploaded as knowledge/context for a custom GPT
that helps with the thesis, documentation, supervisor updates, and implementation
planning for `UnembodiedNavigation`.

Use them together:

| File | Purpose |
| --- | --- |
| [`assistant_instructions.md`](assistant_instructions.md) | Behavioral rules for the custom GPT. |
| [`project_story.md`](project_story.md) | Clean thesis storyline and contribution viewpoint. |
| [`current_setup.md`](current_setup.md) | Active runtime/config/result surface to avoid mixing old and current claims. |
| [`evidence_status.md`](evidence_status.md) | What is current evidence, historical evidence, diagnostic, synthetic, or model-only. |
| [`workflow_rules.md`](workflow_rules.md) | Modular, evidence-first way of working based on old-paper lessons. |
| [`ground_truth_firewall.md`](ground_truth_firewall.md) | Rules for using Gazebo ground truth without making it part of the method. |
| [`contribution_claims.md`](contribution_claims.md) | Claims that are safe to make, claims that need caveats, and claims to avoid. |
| [`literature_anchors.md`](literature_anchors.md) | Literature buckets and why each one supports the thesis. |

When these files conflict with live repo evidence, the live evidence registry
and current runtime contract win:

- [`../experiment_registry.md`](../experiment_registry.md)
- [`../current_runtime_contract.yaml`](../current_runtime_contract.yaml)
- [`../paper_vs_current/README.md`](../paper_vs_current/README.md)

