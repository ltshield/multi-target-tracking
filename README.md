# Multi-Target Tracking Planning + Neural Scoring Pipeline

This README explains the current multi-target tracking codebase pipeline: what each file does, how the simulation works, how to run planner comparisons, how to generate neural-network training data, how to train the per-track scorer, and how this connects to the paper-style MCTS framework we are trying to reproduce and extend.

The short version:

```text
Simulated targets move through a 2D world.
A drone maintains belief tracks over those targets.
Track uncertainty grows when targets are not detected.
Track uncertainty shrinks when targets are detected.
If uncertainty gets too large, a target is considered permanently lost.
Different planners choose which target to pursue next.
MCTS can run in the background while the drone is flying/searching.
A neural network can be trained from MCTS demonstrations to score candidate tracks.
The learned scorer can be used directly or to guide MCTS.
```

---

## 1. Big-picture purpose

This codebase is meant to support research on multi-target tracking with a drone or UAV-like agent. The main problem is:

> Given several moving targets with uncertain positions, which target should the drone pursue next to minimize total future uncertainty and avoid losing targets?

The project currently supports several planner types:

```text
Random planner
Greedy planner
Realtime MCTS planner
Neural per-track scorer planner
Guided MCTS planner using the neural scorer
```

The purpose of the pipeline is to let us compare these planners under the same simulated conditions and evaluate them by:

```text
final uncertainty
average uncertainty
time-integrated uncertainty
number of detections
number of lost targets
distance traveled
selected target sequence
```

---

## 2. Conceptual model

### 2.1 Ground truth vs belief

The simulator separates two things:

```text
Ground truth:
    The target's actual simulated state.
    This is hidden from the planner except through detections.

Belief / track:
    The drone's estimate of where the target is.
    This includes mean state and covariance.
```

Each target state is:

```text
[x, y, vx, vy]
```

where:

```text
x, y   = position
vx, vy = velocity
```

The target's true state is advanced by the `target.py` dynamics. The track belief is advanced and updated by `tracks.py`.

---

### 2.2 Belief uncertainty

Each track has a covariance matrix representing uncertainty in the estimated target state.

When the target is not seen, uncertainty grows through a Kalman-style prediction step:

```text
mean_next = F mean
P_next    = F P F^T + Q
```

where:

```text
F = constant-velocity transition model
P = covariance matrix
Q = process noise matrix
```

When a detection occurs, uncertainty shrinks through a Kalman-style measurement update using the detected position.

---

### 2.3 Lost targets

For our use case, once a target's uncertainty becomes too large, we assume it is no longer recoverable.

That means:

```text
If target uncertainty exceeds the loss threshold:
    mark the track as lost
    remove it from planner action selection
    ignore future detections for that target
    add a large lost-target penalty to metrics
```

This is intentionally conservative. In reality, a drone might later happen across the target, but for this project we assume that once a target is lost, it cannot be recovered.

This assumption makes planner evaluation clearer:

```text
Good planner:
    keeps all targets below the loss threshold

Bad planner:
    repeatedly services easy targets while allowing neglected targets to become lost
```

---

## 3. How the simulation works

The simulation is run by `simulate_run.py`.

At a high level, the simulator loops through:

```text
1. Choose a target to pursue.
2. Fly toward that target's current belief center.
3. Begin an elliptic shifting spiral search around that belief.
4. Continue until:
      selected target is found,
      selected target is lost,
      search times out,
      all targets are lost,
      or the mission budget ends.
5. Choose the next target.
```

During every time step:

```text
true targets move
track beliefs predict forward
uncertainty grows
lost thresholds are checked
the drone moves
detections may occur during search
track beliefs are updated if detected
```

---

### 3.1 Opportunistic detections

The simulator supports opportunistic detections, but these should usually be off for main experiments.

```text
opportunistic_detections = false
```

If opportunistic detections are on, the drone can update tracks when it crosses over targets while simply flying to the selected target. If they are off, detections primarily happen during intentional search.

For cleaner evaluation, use:

```yaml
opportunistic_detections: false
```

Then use opportunistic detections only as an ablation.

---

### 3.2 Search behavior

After flying to a target's belief center, the drone executes an elliptic shifting spiral search.

The idea is:

```text
The target belief is uncertain and moving.
The search pattern should cover the uncertain region around the predicted target state.
The ellipse is shaped by the track covariance.
The search center shifts according to the predicted target velocity.
```

This approximates the paper's low-level coverage/search component while keeping the simulator simple enough to support many planner comparisons.

---

## 4. Relationship to the paper framework

The paper-style framework separates the problem into two layers:

```text
Low-level coverage/search:
    If the drone chooses target i, how likely is it to find it?
    How long does the search take?
    What happens if it finds or misses the target?

High-level planning:
    Which target should the drone pursue next?
    How should the planner reason over future find/miss outcomes?
```

Our code follows that same structure.

### 4.1 Low-level search

The simulator uses an elliptic shifting spiral to represent the low-level search behavior.

The MCTS planner does not simulate every spiral waypoint inside every rollout. Instead, it uses a cheaper coverage estimator:

```text
covered area ≈ sensor width * drone speed * search time + initial sensor footprint
probability of find ≈ function of covered area and covariance area
```

This lets MCTS reason over many future target choices without running a full drone simulation inside every tree rollout.

---

### 4.2 High-level MCTS

The realtime MCTS planner models decisions as:

```text
state
  -> choose target action
      -> find outcome
      -> miss outcome
```

Each action has two possible outcomes:

```text
Find:
    target is detected
    covariance shrinks
    target time_since_seen resets

Miss:
    target is not detected during the useful search window
    uncertainty continues to grow
```

The planner tries to minimize future system uncertainty and avoid lost targets.

---

### 4.3 Realtime planning assumption

The revised MCTS planner is designed to better match real-world operation:

```text
The current target pursuit is fixed while the drone flies/searches.
MCTS plans in the background during the time the drone is executing.
When the target is found or missed/lost, MCTS uses the conditional branch for that outcome.
```

So planning time is no longer treated as a free instantaneous operation. MCTS gets planning computation proportional to how long the current flight/search process takes.

---

## 5. Project extension beyond the paper

The paper framework uses MCTS as the high-level planner. Our project extends this by adding a neural planning layer.

The expansion is:

```text
1. Use the simulator + surrogate tracker to produce belief states.
2. Use MCTS as a teacher to choose targets.
3. Train a neural network to score candidate tracks from the current belief state.
4. Use that neural scorer as:
      a direct learned planner, or
      a prior/guide inside MCTS.
```

This creates two learned-planner variants:

```text
Neural scorer:
    Fast direct policy.
    Scores each active track and chooses the highest scoring target.

Guided MCTS:
    Still performs MCTS planning.
    Uses the neural scorer to bias MCTS toward promising target branches.
```

This is similar in spirit to AlphaGo-style planning:

```text
policy network:
    suggests promising actions

search/MCTS:
    reasons about future consequences

optional future value network:
    estimates final uncertainty from a belief state
```

The current neural pipeline implements the policy/scoring side. A future extension can add a value head that predicts final uncertainty or lost-target risk.

---

## 6. Important files

### 6.1 Core simulation files

#### `target.py`

Represents true target dynamics.

Responsibilities:

```text
stores true target state [x, y, vx, vy]
advances target state using constant velocity
adds optional process noise
returns true target positions for detection
```

The planner does not directly use true target state, except indirectly through sensor detections.

---

#### `drone.py`

Represents the drone.

Responsibilities:

```text
stores drone position, speed, sensor range, and mission budget
moves toward waypoints
tracks elapsed time and distance traveled
detects targets inside the sensor footprint
```

The drone currently has a circular sensor footprint.

---

#### `tracks.py`

Represents belief tracks.

Responsibilities:

```text
stores estimated target state and covariance
predicts belief forward with constant-velocity Kalman dynamics
updates belief after detections
computes uncertainty metrics
marks targets as lost when thresholds are exceeded
```

This is the surrogate tracker component.

---

#### `coverage_spiral.py`

Implements the elliptic shifting spiral search planner.

Responsibilities:

```text
constructs a search ellipse from target covariance
shifts the search center according to target velocity
generates waypoints for spiral coverage
```

This is the low-level search pattern used by the simulator.

---

#### `simulate_run.py`

Runs one full simulation and saves the run history.

Responsibilities:

```text
loads config
loads planner
creates drone, targets, and tracks
runs the simulation loop
updates targets and beliefs
handles detections and lost targets
runs background conditional MCTS if the planner supports it
logs every frame to JSON
```

Output:

```text
runs/some_run.json
```

---

### 6.2 Baseline planner files

#### `random_planner.py` or `planners.RandomPlanner`

Chooses an active target randomly.

Useful as a lower baseline.

---

#### `greedy_planners.py` or `planners.GreedyDistanceAwarePlanner`

Chooses targets using hand-coded heuristic logic.

Common greedy choices include:

```text
most uncertain target
largest log-det covariance
uncertainty adjusted by travel distance
```

The greedy distance-aware planner is currently a strong baseline.

---

#### `mcts_planner_realtime.py`

Realtime conditional MCTS planner.

Responsibilities:

```text
chooses the first/current target
runs MCTS in the background during flight/search
builds conditional recommendations for find/miss outcomes
uses a coverage estimator inside rollouts
penalizes lost targets
minimizes future uncertainty
```

This is the closest planner to the paper-style high-level MCTS process.

---

### 6.3 Comparison/visualization files

#### `compare_planners.py`

Runs several planners across several seeds and generates metrics/plots.

Outputs:

```text
runs/comparison_name/
    planner_seed7.json
    planner_seed8.json
    ...
    summary_metrics.csv
    final_uncertainty_trace.png
    average_uncertainty_trace.png
    auc_uncertainty_trace.png
    detections.png
    distance_traveled.png
    uncertainty_over_time_seed7.png
```

---

#### `visualize_run_pygame.py`

Loads a saved run JSON and replays it in Pygame.

Useful for inspecting:

```text
drone path
target paths
belief ellipses
lost targets
selected target
detections
uncertainty evolution
```

---

### 6.4 Neural scorer training files

#### `track_feature_extractors.py`

Defines modular input representations for the neural network.

Included extractors:

```text
CartesianTrackFeatureExtractor
PolarTrackFeatureExtractor
```

The network can be trained with different environment representations by changing:

```bash
--extractor polar
```

or:

```bash
--extractor cartesian
```

To add a new representation, add a new extractor class that provides:

```python
global_dim
track_dim
build_batch(...)
```

---

#### `scenario_sampler.py`

Generates randomized training scenarios.

It randomizes:

```text
number of targets
target starting distances from drone
target angles/orientations
target speeds
target headings
target process noise
```

The drone parameters stay fixed from the base YAML unless you change them in the base config.

This prevents the neural network from overfitting to one fixed target layout.

---

#### `track_scorer_model.py`

Defines the shared-weight per-track neural scorer.

Architecture:

```text
global features + candidate track features -> scalar score
```

The same network is applied to every track.

This is better than a fixed-slot policy because the scoring rule is tied to track features rather than target identity.

---

#### `train_track_scorer_randomized.py`

Generates MCTS demonstration data and trains the per-track scorer.

Responsibilities:

```text
sample randomized scenarios
run MCTS teacher
collect belief states and MCTS-selected targets
filter out bad teacher episodes
save dataset
train neural scorer
save model
```

Outputs:

```text
data/randomized_track_scorer_demos.npz
models/track_scorer.pt
```

---

#### `track_scorer_planner_modular.py`

Loads the trained model and uses it as a direct planner.

Process:

```text
build features
score each active track
choose highest-scoring track
```

This is the neural scorer planner.

---

#### `guided_track_scorer_mcts_modular.py`

Loads the trained neural scorer and uses it to guide MCTS.

Process:

```text
MCTS still simulates future find/miss outcomes.
The neural scorer biases action/rollout scoring toward promising tracks.
```

This is the guided MCTS planner.

---

## 7. Configuration file structure

A typical YAML config should include:

```yaml
seed: 7
dt: 0.5
mission_budget: 900.0
max_steps: 10000

opportunistic_detections: false

drone:
  initial_position: [0.0, 0.0]
  speed: 30.0
  sensor_range: 50.0
  detection_probability: 1.0

belief:
  acceleration_noise_std: 0.03
  measurement_noise_std: 20.0
  initial_position_std: 75.0
  initial_velocity_std: 3.0
  covariance_scale_for_search: 2.0

search:
  max_search_time_per_decision: 65.0

loss:
  max_position_trace_before_lost: 250000.0
  max_position_logdet_before_lost:
  max_time_since_seen_before_lost:
  lost_target_penalty: 1000000.0

targets:
  - target_id: 1
    initial_state: [500.0, 250.0, 2.0, -0.5]
    process_noise_std: [0.0, 0.0, 0.002, 0.002]
```

For neural training, this base config provides default drone/search/belief/loss settings. The randomized scenario sampler replaces the target list with random target layouts.

---

## 8. Common commands

The commands below assume you are running from the project root.

On Windows PowerShell, use `^` for line continuation. On Mac/Linux, replace `^` with `\`.

---

### 8.1 Run one simulation

Random planner:

```bash
python src/core/simulate_run.py ^
  --config configs/basic_3target.yaml ^
  --planner random_planner.RandomPlanner ^
  --output runs/random_test.json
```

Greedy planner:

```bash
python src/core/simulate_run.py ^
  --config configs/basic_3target.yaml ^
  --planner greedy_planners.GreedyDistanceAwarePlanner ^
  --output runs/greedy_test.json
```

Realtime MCTS:

```bash
python src/core/simulate_run.py ^
  --config configs/basic_3target.yaml ^
  --planner mcts_planner_realtime.MCTSPlanner ^
  --output runs/mcts_test.json
```

Neural scorer:

```bash
python src/core/simulate_run.py ^
  --config configs/basic_3target.yaml ^
  --planner track_scorer_planner_modular.TrackScorerPlanner ^
  --output runs/scorer_test.json
```

Guided MCTS:

```bash
python src/core/simulate_run.py ^
  --config configs/basic_3target.yaml ^
  --planner guided_track_scorer_mcts_modular.GuidedTrackScorerMCTSPlanner ^
  --output runs/guided_test.json
```

---

### 8.2 Visualize a run

```bash
python src/core/visualize_run_pygame.py ^
  --input runs/mcts_test.json
```

Controls:

```text
SPACE  pause / unpause
LEFT   step backward
RIGHT  step forward
UP     faster playback
DOWN   slower playback
R      restart
ESC    quit
```

---

### 8.3 Compare planners

```bash
python src/core/compare_planners.py ^
  --config configs/basic_3target.yaml ^
  --output-dir runs/planner_compare ^
  --seeds 7 8 9 10 11 ^
  --planners random=random_planner.RandomPlanner greedy=greedy_planners.GreedyDistanceAwarePlanner mcts=mcts_planner_realtime.MCTSPlanner
```

Compare all major planners:

```bash
python src/core/compare_planners.py ^
  --config configs/basic_3target.yaml ^
  --output-dir runs/full_compare ^
  --seeds 7 8 9 10 11 ^
  --planners random=random_planner.RandomPlanner greedy=greedy_planners.GreedyDistanceAwarePlanner mcts=mcts_planner_realtime.MCTSPlanner scorer=track_scorer_planner_modular.TrackScorerPlanner guided=guided_track_scorer_mcts_modular.GuidedTrackScorerMCTSPlanner
```

---

### 8.4 Train neural scorer from randomized MCTS demonstrations

Quick smoke test:

```bash
python src/core/train_track_scorer_randomized.py ^
  --base-config configs/basic_3target.yaml ^
  --dataset data/randomized_track_scorer_demos_test.npz ^
  --model-output models/track_scorer_test.pt ^
  --episodes 5 ^
  --teacher-iterations 50 ^
  --epochs 5 ^
  --extractor polar ^
  --max-lost-for-keep 1
```

Serious run:

```bash
python src/core/train_track_scorer_randomized.py ^
  --base-config configs/basic_3target.yaml ^
  --dataset data/randomized_track_scorer_demos.npz ^
  --model-output models/track_scorer.pt ^
  --episodes 500 ^
  --teacher-iterations 750 ^
  --epochs 60 ^
  --extractor polar ^
  --max-lost-for-keep 0
```

If too many episodes are discarded:

```bash
--max-lost-for-keep 1
```

If training is too slow:

```bash
--teacher-iterations 250
--episodes 100
```

---

## 9. Training data filtering

The training file filters bad teacher trajectories.

An episode is kept only if:

```text
num_lost <= max_lost_for_keep
final_metric <= max_final_metric_for_keep
episode_samples >= min_samples_per_episode
```

Default:

```text
max_lost_for_keep = 0
```

This means only episodes where MCTS loses no targets are used for training.

Why this matters:

```text
If MCTS loses 5 of 6 targets, we do not want the neural scorer to imitate that trajectory.
```

During data collection, the script prints:

```text
episode=0007 targets=5 episode_samples=18 total_kept_samples=92 final_metric=18432.12 active=5 lost=0 KEPT
episode=0008 targets=6 episode_samples=14 total_kept_samples=92 final_metric=2000041.55 active=4 lost=2 DISCARDED (num_lost 2 > 0)
```

---

## 10. Recommended workflow

### Step 1: Verify simulation works

```bash
python src/core/simulate_run.py ^
  --config configs/basic_3target.yaml ^
  --planner greedy_planners.GreedyDistanceAwarePlanner ^
  --output runs/greedy_smoke.json
```

Visualize:

```bash
python src/core/pygame_visualize_run.py ^
  --input runs/greedy_smoke.json
```

---

### Step 2: Compare baseline planners

```bash
python src/core/compare_planners.py ^
  --config configs/basic_3target.yaml ^
  --output-dir runs/baseline_compare ^
  --seeds 7 8 9 10 11 ^
  --planners random=random_planner.RandomPlanner greedy=greedy_planners.GreedyDistanceAwarePlanner mcts=mcts_planner_realtime.MCTSPlanner
```

---

### Step 3: Generate MCTS demonstrations and train scorer

Start small:

```bash
python src/core/train_track_scorer_randomized.py ^
  --base-config configs/basic_3target.yaml ^
  --dataset data/demo_small.npz ^
  --model-output models/track_scorer.pt ^
  --episodes 20 ^
  --teacher-iterations 250 ^
  --epochs 10 ^
  --extractor polar ^
  --max-lost-for-keep 1
```

Then scale:

```bash
python src/core/train_track_scorer_randomized.py ^
  --base-config configs/basic_3target.yaml ^
  --dataset data/randomized_track_scorer_demos.npz ^
  --model-output models/track_scorer.pt ^
  --episodes 500 ^
  --teacher-iterations 750 ^
  --epochs 60 ^
  --extractor polar ^
  --max-lost-for-keep 0
```

---

### Step 4: Compare learned planners

```bash
python src/core/compare_planners.py ^
  --config configs/basic_3target.yaml ^
  --output-dir runs/learned_compare ^
  --seeds 7 8 9 10 11 ^
  --planners greedy=greedy_planners.GreedyDistanceAwarePlanner mcts=mcts_planner_realtime.MCTSPlanner scorer=track_scorer_planner_modular.TrackScorerPlanner guided=guided_track_scorer_mcts_modular.GuidedTrackScorerMCTSPlanner
```

---

## 11. How to interpret results

### Good signs

```text
final_uncertainty_trace is low
avg_uncertainty_trace is low
auc_uncertainty_trace is low
num_lost is zero
unique_detected_targets equals the number of targets
selected_track_sequence alternates between targets instead of fixating
```

### Bad signs

```text
uncertainty jumps to ~1e6 or ~2e6
selected_track_sequence repeats one target hundreds of times
unique_detected_targets is less than total targets
neural scorer produces many decisions but loses targets
guided MCTS is worse than unguided MCTS
```

A jump near `1e6` usually means one lost target penalty. A jump near `2e6` usually means two lost target penalties, depending on the configured penalty.

---

## 12. Current known issues and next improvements

### 12.1 Greedy is currently very strong

The greedy distance-aware planner may outperform early MCTS and neural scorers. That is okay. It gives us a meaningful baseline.

---

### 12.2 MCTS teacher stability matters

If MCTS is unstable, the neural scorer will learn unstable behavior. Improve MCTS before generating a large serious dataset.

Useful MCTS tuning knobs:

```text
iterations
iterations_per_second
max_depth
lost_target_penalty
lost_trace_threshold
covariance_scale_for_detection
recent_revisit_penalty_window
```

---

### 12.3 Neural scorer can learn shortcuts

If the scorer repeatedly chooses one target, it may have learned a bad shortcut.

Fixes:

```text
filter bad episodes
increase training diversity
use polar extractor
train on more scenarios
add value head
use sample weights
weaken guided prior weight
```

---

### 12.4 Add a value network

The current neural pipeline trains a policy/scorer network.

A future AlphaGo-style extension would train:

```text
policy head:
    scores candidate tracks

value head:
    predicts final uncertainty / lost-target risk
```

Then guided MCTS can use:

```text
policy head to guide action selection
value head to evaluate leaf states
```

This would reduce reliance on expensive rollouts.

---

## 13. Dependencies

Core simulation:

```bash
pip install numpy scipy matplotlib pyyaml
```

Visualization:

```bash
pip install pygame
```

Neural training:

```bash
pip install torch
```

Optional RL experiments:

```bash
pip install gymnasium stable-baselines3
```

---

## 14. Summary

This pipeline supports a complete research loop:

```text
1. Simulate multi-target tracking with belief uncertainty.
2. Use baseline planners and MCTS to choose targets.
3. Evaluate planners by uncertainty and target loss.
4. Generate randomized MCTS demonstration data.
5. Train a modular per-track neural scorer.
6. Use the scorer directly as a planner.
7. Use the scorer to guide MCTS.
8. Compare learned and non-learned planners across seeds.
```

The core research question this enables is:

> Can a learned neural scorer approximate or improve MCTS target selection, and can it guide MCTS to achieve lower uncertainty or similar uncertainty with less search?

That directly supports the project direction:

```text
surrogate tracker -> belief updates -> neural network planner -> guided MCTS / planner comparison
```
