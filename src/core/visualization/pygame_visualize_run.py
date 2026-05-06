"""Replay a saved multi-target tracking run in Pygame.

This file does not run simulation logic. It only visualizes a JSON run generated
by simulate_run.py.

Example:
    python src/core/visualize_run_pygame.py --input runs/random_run.json

Controls:
    SPACE  pause / unpause
    LEFT   step backward
    RIGHT  step forward
    UP     faster playback
    DOWN   slower playback
    R      restart replay
    ESC    quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pygame


WIDTH = 1200
HEIGHT = 800
FPS = 60

WORLD_SCALE = 0.45
SCREEN_CENTER = np.array([WIDTH / 2, HEIGHT / 2], dtype=float)

COVARIANCE_SCALE_FOR_DRAWING = 1.0

BACKGROUND = (18, 20, 28)
WHITE = (235, 235, 235)
BLUE = (100, 200, 255)
SENSOR_BLUE = (60, 100, 130)
SELECTED_ELLIPSE = (185, 185, 255)
UNSELECTED_ELLIPSE = (90, 90, 130)
SELECTED_CENTER = (230, 230, 255)
UNSELECTED_CENTER = (135, 135, 175)

TARGET_COLORS = {
    "1": (255, 120, 120),
    "2": (120, 255, 160),
    "3": (255, 220, 120),
    "4": (220, 120, 255),
    "5": (120, 220, 255),
}


def world_to_screen(point: np.ndarray) -> tuple[int, int]:
    x = SCREEN_CENTER[0] + point[0] * WORLD_SCALE
    y = SCREEN_CENTER[1] - point[1] * WORLD_SCALE
    return int(x), int(y)


def screen_radius(world_radius: float) -> int:
    return max(1, int(world_radius * WORLD_SCALE))


def as_np(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = WHITE,
) -> None:
    image = font.render(text, True, color)
    surface.blit(image, pos)


def covariance_ellipse_points(
    center: np.ndarray,
    covariance_2x2: np.ndarray,
    scale: float = 1.0,
    num_points: int = 80,
) -> list[tuple[int, int]]:
    cov = 0.5 * (covariance_2x2 + covariance_2x2.T)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-9)
    eigenvectors = eigenvectors[:, order]

    axes = scale * np.sqrt(eigenvalues)

    points = []
    for theta in np.linspace(0, 2.0 * np.pi, num_points):
        local = np.array(
            [
                axes[0] * np.cos(theta),
                axes[1] * np.sin(theta),
            ],
            dtype=float,
        )
        world = center + eigenvectors @ local
        points.append(world_to_screen(world))

    return points


def load_run(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_paths(history: list[dict]) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    """Precompute paths up to each frame for smooth replay."""

    drone_path = []
    target_paths: dict[str, list[np.ndarray]] = {}

    for frame in history:
        drone_path.append(as_np(frame["drone"]["position"]))

        for target_id, target_data in frame["targets"].items():
            target_paths.setdefault(target_id, [])
            target_paths[target_id].append(as_np(target_data["position"]))

    return drone_path, target_paths


def draw_polyline(
    surface: pygame.Surface,
    points: list[np.ndarray],
    color: tuple[int, int, int],
    width: int = 2,
    max_points: int = 1200,
) -> None:
    if len(points) < 2:
        return

    shown = points[-max_points:]
    pygame.draw.lines(
        surface,
        color,
        False,
        [world_to_screen(point) for point in shown],
        width,
    )


def render(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    run: dict,
    frame_idx: int,
    paused: bool,
    playback_speed: int,
    drone_path: list[np.ndarray],
    target_paths: dict[str, list[np.ndarray]],
) -> None:
    screen.fill(BACKGROUND)

    history = run["history"]
    frame = history[frame_idx]

    # Paths up to current frame.
    draw_polyline(screen, drone_path[: frame_idx + 1], BLUE, width=2)

    for target_id, path in target_paths.items():
        color = TARGET_COLORS.get(target_id, (220, 220, 220))
        draw_polyline(screen, path[: frame_idx + 1], color, width=1)

    selected_track_id = frame["selected_track_id"]
    selected_track_id_str = None if selected_track_id is None else str(selected_track_id)

    # Draw belief ellipses.
    for track_id, track_data in frame["tracks"].items():
        is_lost = bool(track_data.get("is_lost", False))
        if is_lost:
            center = as_np(track_data["position"])
            pygame.draw.circle(screen, (120, 120, 120), world_to_screen(center), 8, 2)
            label_pos = world_to_screen(center + np.array([15.0, 15.0]))
            draw_text(
                screen,
                small_font,
                f"T{track_id} LOST",
                label_pos,
                (160, 160, 160),
            )
            continue
        center = as_np(track_data["position"])
        cov = as_np(track_data["position_covariance"])

        is_selected = track_id == selected_track_id_str

        ellipse_color = SELECTED_ELLIPSE if is_selected else UNSELECTED_ELLIPSE
        center_color = SELECTED_CENTER if is_selected else UNSELECTED_CENTER

        ellipse_points = covariance_ellipse_points(
            center=center,
            covariance_2x2=cov,
            scale=COVARIANCE_SCALE_FOR_DRAWING,
        )

        if len(ellipse_points) >= 3:
            pygame.draw.lines(
                screen,
                ellipse_color,
                True,
                ellipse_points,
                2 if is_selected else 1,
            )

        pygame.draw.circle(screen, center_color, world_to_screen(center), 5)

        label_pos = world_to_screen(center + np.array([15.0, 15.0]))
        draw_text(
            screen,
            small_font,
            f"T{track_id} tr={track_data['position_variance_trace']:.0f}",
            label_pos,
            center_color,
        )

    # Draw true targets.
    for target_id, target_data in frame["targets"].items():
        position = as_np(target_data["position"])
        color = TARGET_COLORS.get(target_id, (255, 255, 255))
        pygame.draw.circle(screen, color, world_to_screen(position), 7)

    # Draw drone and sensor range.
    drone_position = as_np(frame["drone"]["position"])
    drone_screen = world_to_screen(drone_position)
    pygame.draw.circle(screen, BLUE, drone_screen, 8)

    pygame.draw.circle(
        screen,
        SENSOR_BLUE,
        drone_screen,
        screen_radius(frame["drone"]["sensor_range"]),
        1,
    )

    # Detection flash.
    if frame["detections"]:
        pygame.draw.circle(
            screen,
            WHITE,
            drone_screen,
            screen_radius(frame["drone"]["sensor_range"]),
            3,
        )

    # Draw current spiral waypoint if available.
    if "spiral_waypoint" in frame:
        waypoint = as_np(frame["spiral_waypoint"])
        pygame.draw.circle(screen, (80, 140, 220), world_to_screen(waypoint), 4)

    # HUD.
    metadata = run.get("metadata", {})
    config = run.get("config", {})
    hud_lines = [
        f"Planner: {metadata.get('planner', 'unknown')}",
        f"Frame: {frame_idx + 1} / {len(history)}",
        f"Playback speed: {playback_speed} frame(s)/tick",
        f"Paused: {paused}",
        f"Event: {frame['event']}",
        f"Mode: {frame['mode']}",
        f"Selected track: {frame['selected_track_id']}",
        f"Detected: {frame['detections']}",
        f"Time: {frame['time']:.1f} / {config.get('mission_budget', 0):.1f}",
        f"Remaining: {frame['remaining_budget']:.1f}",
        f"Total trace: {frame['metrics']['total_position_trace']:.1f}",
        f"Opportunistic detections: {config.get('opportunistic_detections')}",
        f"Active: {frame['metrics'].get('num_active')}",
        f"Lost: {frame['metrics'].get('num_lost')}",
        f"Newly lost: {frame.get('newly_lost_tracks', [])}",
        "SPACE pause | LEFT/RIGHT step | UP/DOWN speed | R restart | ESC quit",
    ]

    x, y = 20, 20
    for line in hud_lines:
        draw_text(screen, font, line, (x, y))
        y += 25

    # Legend.
    legend_y = HEIGHT - 105
    draw_text(screen, small_font, "Legend:", (20, legend_y))
    draw_text(screen, small_font, "Filled colored dots = true targets", (20, legend_y + 22))
    draw_text(screen, small_font, "Ellipses = track belief uncertainty", (20, legend_y + 44))
    draw_text(screen, small_font, "Blue circle = drone sensor footprint", (20, legend_y + 66))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="runs/random_run.json")
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--speed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run = load_run(Path(args.input))
    history = run["history"]

    if not history:
        raise ValueError("Run contains no history frames.")

    drone_path, target_paths = build_paths(history)

    pygame.init()
    pygame.display.set_caption("MTT Run Replay")

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 19)
    small_font = pygame.font.SysFont("consolas", 15)

    frame_idx = 0
    paused = bool(args.start_paused)
    playback_speed = max(1, int(args.speed))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_RIGHT:
                    frame_idx = min(len(history) - 1, frame_idx + 1)
                    paused = True

                elif event.key == pygame.K_LEFT:
                    frame_idx = max(0, frame_idx - 1)
                    paused = True

                elif event.key == pygame.K_UP:
                    playback_speed = min(50, playback_speed + 1)

                elif event.key == pygame.K_DOWN:
                    playback_speed = max(1, playback_speed - 1)

                elif event.key == pygame.K_r:
                    frame_idx = 0
                    paused = False

        if not paused:
            frame_idx = min(len(history) - 1, frame_idx + playback_speed)

            if frame_idx >= len(history) - 1:
                paused = True

        render(
            screen=screen,
            font=font,
            small_font=small_font,
            run=run,
            frame_idx=frame_idx,
            paused=paused,
            playback_speed=playback_speed,
            drone_path=drone_path,
            target_paths=target_paths,
        )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()