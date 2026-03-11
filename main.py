import time
import threading

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from behaviors.consensus_algorithm import ConsensusAlgorithm
from behaviors.collision_avoidance_algorithm import CollisionAvoidanceAlgorithm
from behaviors.formation_control_algorithm import FormationControlAlgorithm
from visualizer import DroneSwarmVisualizer
from drone import Drone
from drone_state import DroneState
from swarm import Swarm


# ---------------------------------------------------------------------------
# Formation simulation (original)
# ---------------------------------------------------------------------------

class DroneSwarmApp:
    def __init__(self, parent):
        """
        Initialize the Drone Swarm Simulation application inside *parent*
        (which may be a Tk root window or any Frame-like widget).
        """
        self.root = parent

        self.target_point = np.array([0, 0, 0])  # Initial target point
        self.is_x_at_origin = True  # State to track if the target is at the origin

        # Simulation parameters
        self.num_drones = 100  # Number of drones in the swarm
        self.iterations = 100  # Number of iterations (not currently used)
        self.epsilon = 0.1  # Parameter for the consensus algorithm
        self.collision_threshold = 1.0  # Minimum distance to avoid collisions
        self.interval = 200  # Time interval between simulation updates (ms)

        # UI control variables
        self.formation_type = tk.StringVar(value="line")  # Formation type selection
        self.zoom_level = tk.DoubleVar(value=10.0)  # Zoom level for visualization

        # Define behavior algorithms
        self.behavior_algorithms = [
            ConsensusAlgorithm(self.epsilon),
            CollisionAvoidanceAlgorithm(self.collision_threshold),
            FormationControlAlgorithm(self.formation_type.get())
        ]

        # Initialize the swarm with 3D random positions
        self.drones = [Drone(np.random.rand(3) * 10, i) for i in range(self.num_drones)]

        # Initialize the visualizer
        self.visualizer = DroneSwarmVisualizer(self.drones, self.formation_type.get())

        # Set up the UI
        self.setup_ui()

        # Simulation state
        self.running = False

    def setup_ui(self):
        """
        Set up the graphical user interface.
        """
        # Create a frame for controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Formation selection radio buttons
        ttk.Label(control_frame, text="Formation:").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Line", variable=self.formation_type, value="line", command=self.update_formation).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Circle", variable=self.formation_type, value="circle", command=self.update_formation).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Square", variable=self.formation_type, value="square", command=self.update_formation).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Random", variable=self.formation_type, value="random", command=self.update_formation).pack(anchor=tk.W)

        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Zoom level control
        ttk.Label(control_frame, text="Zoom Level:").pack(anchor=tk.W)
        zoom_scale = ttk.Scale(control_frame, from_=5.0, to=20.0, orient=tk.HORIZONTAL, variable=self.zoom_level, command=self.update_zoom)
        zoom_scale.pack(anchor=tk.W)

        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Color mode control
        ttk.Label(control_frame, text="Color Mode:").pack(anchor=tk.W)
        self.color_mode = tk.StringVar(value="by_index")
        ttk.Radiobutton(control_frame, text="By Index", variable=self.color_mode, value="by_index", command=self.update_color_mode).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="By Distance", variable=self.color_mode, value="by_distance", command=self.update_color_mode).pack(anchor=tk.W)

        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Button to change the target position
        ttk.Button(control_frame, text="Change X Position", command=self.change_x_position).pack(pady=10)

        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Animate", command=self.toggle_simulation)
        self.start_button.pack(pady=10)

        # Canvas to display the swarm visualization
        self.canvas = FigureCanvasTkAgg(self.visualizer.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def update_formation(self):
        """
        Update the formation control algorithm when the user selects a different formation.
        """
        self.behavior_algorithms[-1] = FormationControlAlgorithm(self.formation_type.get())
        self.visualizer.formation_type = self.formation_type.get()
        self.canvas.draw()

    def update_zoom(self, event):
        """
        Update the visualization zoom level when the user adjusts the zoom slider.
        """
        self.visualizer.update_zoom(self.zoom_level.get())
        self.canvas.draw()

    def update_color_mode(self):
        """
        Update the color mode of the drones in the visualization.
        """
        self.visualizer.color_mode = self.color_mode.get()
        self.visualizer.update_colors()
        self.canvas.draw()

    def toggle_simulation(self):
        """
        Start or stop the simulation when the button is clicked.
        """
        if self.running:
            self.running = False
            self.start_button.config(text="Animate")
        else:
            self.running = True
            self.start_button.config(text="Stop")
            threading.Thread(target=self.run_simulation).start()

    def change_x_position(self):
        """
        Toggle the target position between [20, 0, 0] and [0, 0, 0].
        """
        if self.is_x_at_origin:
            self.target_point = np.array([20, 0, 0])
        else:
            self.target_point = np.array([0, 0, 0])

        self.is_x_at_origin = not self.is_x_at_origin
        self.update_target_positions()

    def update_target_positions(self):
        """
        Update the target positions of the drones based on the current formation.
        """
        formation = self.behavior_algorithms[-1].get_formation(self.drones)
        for drone, target in zip(self.drones, formation):
            drone.target_position = self.target_point + target

        # Update the target point in the formation control algorithm
        self.behavior_algorithms[-1].set_target_point(self.target_point)

    def run_simulation(self):
        """
        Run the simulation loop, updating drone positions and refreshing the visualization.
        """
        while self.running:
            # Update each drone's position based on behavior algorithms
            for drone in self.drones:
                neighbor_positions = [other_drone.communicate() for other_drone in self.drones if other_drone != drone]
                drone.update_position(neighbor_positions, self.behavior_algorithms)

            # Update the view to follow the drones
            self.visualizer.update_view(self.drones)

            # Refresh visualization
            self.visualizer.update()
            self.canvas.draw()


# ---------------------------------------------------------------------------
# Mission simulation (new — leader election + sector mapping)
# ---------------------------------------------------------------------------

class MissionSwarmApp:
    """
    Self-healing drone swarm mission simulation.

    Features:
    - Start / stop a reconnaissance mission on a 2-D grid.
    - Visual mission map updated in real time.
    - Drone status panel (active drones, current leader, map coverage).
    - Kill-leader button to demonstrate fault tolerance.
    - Bully-algorithm leader election with event log.
    - Export final mission map as mission_map.jpg.
    """

    _UPDATE_MS = 200  # UI refresh interval in milliseconds

    def __init__(self, parent: tk.Widget) -> None:
        self.frame = parent
        self.swarm: Swarm | None = None
        self.sim_thread: threading.Thread | None = None
        self._last_log_len = 0
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        left = ttk.Frame(self.frame, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        left.pack_propagate(False)

        right = ttk.Frame(self.frame)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ── Status section ──────────────────────────────────────────────
        ttk.Label(left, text="Mission Status", font=("Arial", 11, "bold")).pack(
            anchor=tk.W, pady=(4, 2)
        )
        self._status_vars = {
            "Active Drones": tk.StringVar(value="—"),
            "Leader": tk.StringVar(value="—"),
            "Mission": tk.StringVar(value="Standby"),
            "Map Coverage": tk.StringVar(value="0 %"),
        }
        for label, var in self._status_vars.items():
            row = ttk.Frame(left)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label}:", width=14, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=var, foreground="#0055cc").pack(side=tk.LEFT)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Drone count ─────────────────────────────────────────────────
        ttk.Label(left, text="Number of Drones:").pack(anchor=tk.W)
        self._num_drones_var = tk.IntVar(value=50)
        ttk.Scale(
            left, from_=5, to=100, variable=self._num_drones_var, orient=tk.HORIZONTAL
        ).pack(fill=tk.X)
        ttk.Label(left, textvariable=self._num_drones_var).pack(anchor=tk.W)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Buttons ─────────────────────────────────────────────────────
        self._start_btn = ttk.Button(
            left, text="Start Mission", command=self._toggle_mission
        )
        self._start_btn.pack(fill=tk.X, pady=3)

        self._kill_btn = ttk.Button(
            left, text="Kill Leader", command=self._kill_leader, state=tk.DISABLED
        )
        self._kill_btn.pack(fill=tk.X, pady=3)

        self._export_btn = ttk.Button(
            left, text="Export Map", command=self._export_map, state=tk.DISABLED
        )
        self._export_btn.pack(fill=tk.X, pady=3)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Event log ───────────────────────────────────────────────────
        ttk.Label(left, text="Event Log:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        log_frame = ttk.Frame(left)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self._log_text = tk.Text(
            log_frame,
            height=15,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Courier", 8),
            background="#1e1e1e",
            foreground="#d4d4d4",
        )
        log_scroll = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Mission map ─────────────────────────────────────────────────
        self._fig, self._ax = plt.subplots(figsize=(7, 7))
        self._fig.patch.set_facecolor("#f5f5f5")
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_map()

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _toggle_mission(self) -> None:
        if self.swarm and self.swarm.running:
            self.swarm.stop()
            self._start_btn.config(text="Start Mission")
            self._kill_btn.config(state=tk.DISABLED)
            return

        n = int(self._num_drones_var.get())
        self.swarm = Swarm(num_drones=n)
        self.swarm.start()

        self._start_btn.config(text="Stop Mission")
        self._kill_btn.config(state=tk.NORMAL)
        self._export_btn.config(state=tk.NORMAL)
        self._last_log_len = 0

        self.sim_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.sim_thread.start()

        self.frame.after(self._UPDATE_MS, self._tick_ui)

    def _kill_leader(self) -> None:
        if self.swarm:
            self.swarm.kill_leader()

    def _export_map(self) -> None:
        if self.swarm:
            self.swarm.export_map("mission_map.jpg")

    # ------------------------------------------------------------------
    # Simulation loop (background thread)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        from swarm import STEP_SLEEP
        while self.swarm and self.swarm.running:
            self.swarm.step()
            time.sleep(STEP_SLEEP)

    # ------------------------------------------------------------------
    # UI update cycle (main thread, scheduled with after())
    # ------------------------------------------------------------------

    def _tick_ui(self) -> None:
        if self.swarm is None:
            return

        leader = self.swarm.leader
        self._status_vars["Active Drones"].set(str(self.swarm.alive_count))
        self._status_vars["Leader"].set(
            f"Drone {leader.drone_id}" if leader else "None (electing…)"
        )
        self._status_vars["Mission"].set(
            "Sector Mapping" if self.swarm.running else "Complete"
        )
        self._status_vars["Map Coverage"].set(
            f"{self.swarm.scan_progress * 100:.1f} %"
        )

        self._refresh_log()
        self._draw_map()

        if self.swarm.running:
            self.frame.after(self._UPDATE_MS, self._tick_ui)
        else:
            self._start_btn.config(text="Start Mission")
            self._kill_btn.config(state=tk.DISABLED)

    def _refresh_log(self) -> None:
        """Append only newly added log entries to the text widget."""
        if self.swarm is None:
            return
        log = self.swarm.event_log
        new_entries = log[self._last_log_len:]
        if not new_entries:
            return
        self._last_log_len = len(log)
        self._log_text.config(state=tk.NORMAL)
        for entry in new_entries:
            self._log_text.insert(tk.END, entry + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Map drawing
    # ------------------------------------------------------------------

    def _draw_empty_map(self) -> None:
        self._ax.clear()
        self._ax.set_title("Mission Map — awaiting launch", fontsize=12)
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._canvas.draw()

    def _draw_map(self) -> None:
        if self.swarm is None:
            return

        self._ax.clear()

        # Scanned-sector heat-map
        self._ax.imshow(
            self.swarm.grid.T,
            cmap="Blues",
            origin="lower",
            extent=[0, self.swarm.grid_size, 0, self.swarm.grid_size],
            aspect="auto",
            alpha=0.75,
            vmin=0,
            vmax=1,
        )

        # Drone markers
        followers_x, followers_y = [], []
        for drone in self.swarm.drones:
            if not drone.is_alive:
                continue
            x, y = float(drone.position[0]), float(drone.position[1])
            if drone.state == DroneState.LEADER:
                self._ax.scatter(
                    x, y, c="red", s=160, marker="*", zorder=6,
                    label="Leader", edgecolors="darkred", linewidths=0.8,
                )
            elif drone.state == DroneState.CANDIDATE:
                self._ax.scatter(
                    x, y, c="yellow", s=90, marker="^", zorder=5,
                    label="Candidate", edgecolors="orange", linewidths=0.8,
                )
            else:
                followers_x.append(x)
                followers_y.append(y)

        if followers_x:
            self._ax.scatter(
                followers_x, followers_y, c="white", s=25, marker="o",
                zorder=4, alpha=0.85, label="Follower",
                edgecolors="#555", linewidths=0.4,
            )

        # De-duplicated legend
        handles, labels = self._ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            self._ax.legend(
                by_label.values(), by_label.keys(), loc="upper right", fontsize=8
            )

        coverage = self.swarm.scan_progress * 100
        self._ax.set_title(f"Mission Map — {coverage:.1f} % scanned", fontsize=12)
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_xlim(0, self.swarm.grid_size)
        self._ax.set_ylim(0, self.swarm.grid_size)

        self._canvas.draw()


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

# Main entry point for the application
def main():
    root = tk.Tk()
    root.title("Drone Swarm Simulation")
    root.geometry("1280x800")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    formation_frame = ttk.Frame(notebook)
    notebook.add(formation_frame, text="Formation Simulation")

    mission_frame = ttk.Frame(notebook)
    notebook.add(mission_frame, text="Mission Simulation")

    DroneSwarmApp(formation_frame)
    MissionSwarmApp(mission_frame)

    root.mainloop()

if __name__ == "__main__":
    main()
