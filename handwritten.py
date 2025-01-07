import numpy as np
from math import comb
from scipy.integrate import solve_ivp

class Handwriting:
    def __init__(
        self,
        base_speed=1.0,         # Base drawing speed (units/sec)
        mass=1.0,               # Mass of the "pen"
        damping=0.9,            # Linear damping coefficient
        min_duration=0.2,       # Minimum stroke duration
        steps_per_unit=50,      # Number of time steps per unit distance
        max_force=-15.0,        # Maximum force magnitude; negative = unconstrained
        max_force_change=-1,    # Maximum change in force between consecutive steps; negative = unconstrained
        debug=False,            # Whether to print debug info
        verify=True,            # Whether to verify the plan by simulating
        tremor_channels=1,      # Number of tremor channels
        tremor_params=None      # Parameters for each tremor channel
    ):  
        self.base_speed = base_speed
        self.mass = mass
        self.damping = damping
        self.min_duration = min_duration
        self.steps_per_unit = steps_per_unit
        self.max_force = max_force
        self.max_force_change = max_force_change
        self.debug = debug
        self.verify = verify
        self.tremor_channels = tremor_channels
        self.tremor_params = tremor_params if tremor_params is not None else [
            {'max': 60, 'prob': 0.9, 'max_length': 10} for _ in range(tremor_channels)
        ]
        self.current_tremors = [None] * self.tremor_channels

    def _add_tremor(self, N):
        tremor_forces = np.zeros((N, 2))
        for channel in range(self.tremor_channels):
            tremor_max = self.tremor_params[channel]['max']
            tremor_prob = self.tremor_params[channel]['prob']
            tremor_max_length = self.tremor_params[channel]['max_length']
            i = 0
            while i < N:
                if self.current_tremors[channel] is None and np.random.random() < tremor_prob:
                    # Start new tremor
                    tremor_length = np.random.randint(
                        max(1, tremor_max_length // 5),
                        tremor_max_length + 1
                    )
                    # CHANGED to ensure always positive magnitude
                    magnitude = np.random.uniform(0, tremor_max)

                    direction = np.random.rand(2)
                    direction /= np.linalg.norm(direction)

                    # Create tremor profile (ramp up and down between 0 -> 1 -> 0)
                    half_length = tremor_length // 2
                    ramp_up = np.linspace(0, 1, half_length, endpoint=False)
                    ramp_down = np.linspace(1, 0, tremor_length - half_length)
                    profile = np.concatenate([ramp_up, ramp_down])

                    self.current_tremors[channel] = {
                        'remaining': tremor_length,
                        'force': magnitude * direction,
                        'profile': profile
                    }

                if self.current_tremors[channel] is not None:
                    idx = len(self.current_tremors[channel]['profile']) - self.current_tremors[channel]['remaining']
                    if idx < len(self.current_tremors[channel]['profile']):
                        tremor_forces[i] += (
                            self.current_tremors[channel]['force']
                            * self.current_tremors[channel]['profile'][idx]
                        )
                    self.current_tremors[channel]['remaining'] -= 1
                    if self.current_tremors[channel]['remaining'] <= 0:
                        self.current_tremors[channel] = None

                i += 1
        return tremor_forces

    def _bezier_point(self, t, control_points):
        n = len(control_points) - 1
        point = np.zeros(2, dtype=float)
        for i, p in enumerate(control_points):
            point += comb(n, i) * (1 - t)**(n - i) * t**i * p
        return point

    def _approx_curve_length(self, points):
        pts = np.array(points, dtype=float)
        if len(pts) == 2:
            return np.linalg.norm(pts[1] - pts[0])
        else:
            fine_t = np.linspace(0, 1, 20)
            sampled = [self._bezier_point(t, pts) for t in fine_t]
            diffs = np.diff(sampled, axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1)
            return seg_lens.sum()

    def _calculate_stroke_duration(self, points):
        length = self._approx_curve_length(points)
        raw_time = length / self.base_speed
        duration = max(raw_time, self.min_duration)
        if self.debug:
            print(f"[DEBUG] Stroke length={length:.4f}, planned duration={duration:.4f}")
        return duration

    def _desired_path(self, alpha, points):
        pts = np.array(points, dtype=float)
        if len(pts) == 2:
            return pts[0] + alpha * (pts[1] - pts[0])
        else:
            return self._bezier_point(alpha, pts)

    def _make_feedforward_plan(self, points, duration):
        N = max(50, int(duration * self.steps_per_unit))
        t_eval = np.linspace(0, duration, N)

        pos_array = np.zeros((N, 2))
        vel_array = np.zeros((N, 2))
        acc_array = np.zeros((N, 2))
        force_array = np.zeros((N, 2))

        # Generate tremor forces for the entire stroke
        tremor_forces = self._add_tremor(N)

        pos_array[0] = self._desired_path(0, points)
        dt = t_eval[1] - t_eval[0]

        for i in range(1, N):
            next_alpha = i / (N - 1)
            proposed_pos = self._desired_path(next_alpha, points)

            vel = (proposed_pos - pos_array[i - 1]) / dt
            acc = (vel - vel_array[i - 1]) / dt

            required_force = self.mass * acc + self.damping * vel + tremor_forces[i]

            if self.max_force > 0:
                force_mag = np.linalg.norm(required_force)
                if force_mag > self.max_force:
                    scale = self.max_force / force_mag
                    required_force *= scale

            if self.max_force_change > 0 and i > 0:
                force_change = required_force - force_array[i - 1]
                change_mag = np.linalg.norm(force_change)
                if change_mag > self.max_force_change:
                    force_array[i] = force_array[i - 1] + (force_change / change_mag) * self.max_force_change
                else:
                    force_array[i] = required_force
            else:
                force_array[i] = required_force

            acc_array[i] = force_array[i] / self.mass - (self.damping / self.mass) * vel_array[i - 1]
            vel_array[i] = vel_array[i - 1] + acc_array[i] * dt
            pos_array[i] = pos_array[i - 1] + vel_array[i] * dt

        return t_eval, pos_array, vel_array, force_array

    def _verify_feedforward(self, t_eval, pos_array, vel_array, force_array):
        def force_func(t):
            fx = np.interp(t, t_eval, force_array[:, 0])
            fy = np.interp(t, t_eval, force_array[:, 1])
            return np.array([fx, fy], dtype=float)

        def dynamics(t, state):
            px, py, vx, vy = state
            f = force_func(t)
            ax = f[0] / self.mass
            ay = f[1] / self.mass
            return [vx, vy, ax, ay]

        init_state = [
            pos_array[0, 0],
            pos_array[0, 1],
            vel_array[0, 0],
            vel_array[0, 1],
        ]

        sol = solve_ivp(
            dynamics,
            (t_eval[0], t_eval[-1]),
            init_state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-6
        )

        if self.debug:
            final_planned = pos_array[-1]
            final_sim = sol.y[:2, -1]
            diff = final_planned - final_sim
            print("[DEBUG] Verification result:")
            print(f"  Planned final:  {final_planned}")
            print(f"  Simulated final: {final_sim}")
            print(f"  Difference:      {diff}")

    def apply_handwriting(self, curves):
        all_results = []
        for idx, curve in enumerate(curves):
            if self.debug:
                print(f"\n[DEBUG] ---- Stroke {idx} ----")

            duration = self._calculate_stroke_duration(curve)
            t_eval, pos_array, vel_array, force_array = self._make_feedforward_plan(curve, duration)

            if self.verify:
                self._verify_feedforward(t_eval, pos_array, vel_array, force_array)

            stroke_positions = [tuple(p) for p in pos_array]
            all_results.append(stroke_positions)

        return all_results
