# Four-camera warehouse walkthrough

The presentation is built directly in the numbered folders of
`full_story_walkthrough/`. There is no separate result pack.

The intended visual sequence is deliberately simple:

1. The enlarged warehouse and four live views.
2. Four day-zero camera priors.
3. One real Camera C trajectory and the uncertainty-aware GP records collected on it.
4. The same Camera C records fitted with each GP method, shown side by side.
5. Four source-specific GP updates, rather than one pooled field.
6. The camera overlay, overlap region, and source-switching behaviour.
7. The target A→B robot operation: choose a trusted camera source, update belief, and replan if the source changes.

Only Slide 3 is currently being revised. Its source figure is
[`03_uncertainty_aware_collection/figures/camera_c_original_uncertainty_aware_collection.png`](full_story_walkthrough/03_uncertainty_aware_collection/figures/camera_c_original_uncertainty_aware_collection.png).

The former protocol-picture assets have been removed. No visual should be
presented as a measured result unless its source data is named explicitly.
