# Submission validation

Checked on October 17, 2026 (Hong Kong), using the recorded local environments.

- All Python package/script/test files compile.
- The `fruitbowl` wheel builds from `pyproject.toml` without bundling datasets or model files.
- Twelve CPU unit/evidence tests pass (`python -m unittest discover -s tests -v`). They cover hand gaps, grasp/release events, gripper-yaw symmetry, layout fitting, target parsing, model interfaces, saved source-ablation totals and the tie-aware AUC correction.
- The GPU task smoke test renders three scenes, verifies unsolved starts, then verifies successful target drops in all three.
- The released FiLM weights load and run without the demonstration dataset. A three-episode random-layout smoke check succeeds on all three episodes with no wrong-fruit placement. This is a packaging check, not a new performance estimate or a replacement for the historical 150-episode records.
- The 22-slide PowerPoint passes the Office XML validator, exports to PDF, and revised result/diagnostic/teleoperation slides are visually checked. The missing-phone-screenshot placeholder is removed.
- Saved world-model outcome scores reveal a tied-rank AUC bug: the corrected AUC is 0.9541 rather than the original 0.9650. Predictions and 89.36% outcome accuracy are unchanged. The original result and the separate correction audit are both retained.

The historical training runs, every 150-episode evaluation, the full raw-video processing pipeline and the three-arm replay study are not rerun as part of this submission check. Those results are attributed to their preserved development records. Fresh installation on a different machine is not claimed to have been tested.
