# Modular randomized per-track scorer pipeline

This upgrade makes the neural input representation modular and adds randomized
scenario generation for training data.

Files:
- track_feature_extractors.py
- scenario_sampler.py
- track_scorer_model.py
- train_track_scorer_randomized.py
- track_scorer_planner_modular.py

Main idea:
    global features + one candidate track's features -> scalar score

To add a new representation:
1. Add a new extractor class in track_feature_extractors.py.
2. Give it global_dim, track_dim, and build_batch(...).
3. Train with --extractor module.ClassName or add a short alias.

Recommended training:
    python src/core/train_track_scorer_randomized.py ^
      --base-config configs/basic_3target.yaml ^
      --dataset data/randomized_track_scorer_demos.npz ^
      --model-output models/track_scorer.pt ^
      --episodes 500 ^
      --extractor polar ^
      --min-targets 2 ^
      --max-targets 6 ^
      --min-radius 250 ^
      --max-radius 900

Run learned planner:
    python src/core/simulate_run.py ^
      --config configs/basic_3target.yaml ^
      --planner track_scorer_planner_modular.TrackScorerPlanner ^
      --output runs/track_scorer_modular.json

Notes:
- The base YAML supplies drone parameters and default belief/search/loss settings.
- scenario_sampler.py randomizes target count, locations, headings, speeds, and process noise.
- The drone parameters remain fixed unless you change them in the base YAML.
