import numpy as np
import matplotlib.pyplot as plt
import random
from math import comb, pi, sin, cos
from scipy.integrate import solve_ivp

# Import the Handwriting class from the existing file
from handwritten import Handwriting

##########################################################
# Base Classes
##########################################################

class BezierCurve:
    def __init__(self, control_points):
        self.control_points = np.array(control_points)

    def evaluate(self, t):
        """Evaluate the Bezier curve at parameter t (0 <= t <= 1)."""
        n = len(self.control_points) - 1
        point = np.zeros(2, dtype=float)
        for i, p in enumerate(self.control_points):
            point += comb(n, i) * (1 - t)**(n - i) * t**i * p
        return point


class LineSegment:
    def __init__(self, start_point, end_point):
        # Debug print for line segment creation (disabled by default)
        # print(f"[DEBUG] Creating LineSegment from {start_point} to {end_point}")
        self.start_point = np.array(start_point, dtype=float)
        self.end_point = np.array(end_point, dtype=float)

    def evaluate(self, t):
        """Evaluate the line segment at parameter t (0 <= t <= 1)."""
        return (1 - t) * self.start_point + t * self.end_point


##########################################################
# Graph Generation
##########################################################

class GraphGenerator:
    """
    Generates random data for a variety of charts (bar, scatter, pie), adding
    small random noise (based on non_idealness) to simulate imperfection.
    """

    def __init__(self, spacing_noise=0.1, scale_noise=0.1, tick_noise=0.05):
        """
        :param spacing_noise: Controls noise in spacing between elements.
        :param scale_noise: Controls noise in the scale of elements.
        :param tick_noise: Controls noise in tick sizes and labels.
        """
        self.spacing_noise = spacing_noise
        self.scale_noise = scale_noise
        self.tick_noise = tick_noise

    def _add_noise(self, value, noise_level):
        """Add noise to a value based on the specified noise level."""
        return value + random.uniform(-noise_level, noise_level)

    def generate_bar_graph_data(self, num_bars=5):
        """
        Generate random bar graph data with noise.
        Returns:
          - A list of x-positions for the bars.
          - A list of corresponding bar heights.
        """
        bar_heights = [random.uniform(1, 8) for _ in range(num_bars)]
        bar_positions = range(1, num_bars + 1)

        # Add noise to bar positions and heights
        noisy_positions = [self._add_noise(pos, self.spacing_noise) for pos in bar_positions]
        noisy_heights = [self._add_noise(height, self.scale_noise) for height in bar_heights]

        return noisy_positions, noisy_heights

    def generate_scatterplot_data(self, num_points=10):
        """
        Generate random scatterplot data with noise.
        Returns:
          - A list of x-values.
          - A list of corresponding y-values.
        """
        x_values = [random.uniform(0, 10) for _ in range(num_points)]
        y_values = [random.uniform(0, 10) for _ in range(num_points)]

        noisy_x = [self._add_noise(val, self.spacing_noise) for val in x_values]
        noisy_y = [self._add_noise(val, self.spacing_noise) for val in y_values]

        return noisy_x, noisy_y

    def generate_pie_data(self, num_slices=5):
        """
        Generate random pie chart data with noise.
        Returns:
          - A list of slice sizes (they sum to 1.0).
        """
        slices = np.random.rand(num_slices)
        slices /= slices.sum()  # normalize to sum=1
        # Optionally add slight noise per slice if non_idealness is used
        # but keep them normalized so the total remains 1
        noisy_slices = []
        for s in slices:
            s_noisy = self._add_noise(s, self.scale_noise) * 0.01
            noisy_slices.append(s + s_noisy)

        # Re-normalize after the small noise
        total = sum(noisy_slices)
        if total == 0:
            # in a pathological (though unlikely) case all random are zero
            # fallback to uniform distribution
            return [1.0 / num_slices] * num_slices
        else:
            noisy_slices = [s / total for s in noisy_slices]

        return noisy_slices

    def generate_labels(self, label_type='number', count=10, error_rate=0.0):
        """
        Generate labels for graphs with optional error rate.
        :param label_type: 'number' or 'alphabet'
        :param count: Number of labels to generate
        :param error_rate: Probability of a label being incorrect (0.0 to 1.0)
        Returns:
            A list of label strings.
        """
        labels = []
        if label_type == 'number':
            for i in range(1, count + 1):
                if random.random() < error_rate:
                    # Introduce an error by shifting the number
                    faulty_label = str((i + random.choice([-1, 1])) % 10)
                    labels.append(faulty_label)
                else:
                    labels.append(str(i))
        elif label_type == 'alphabet':
            for i in range(count):
                if random.random() < error_rate:
                    # Introduce an error by shifting the letter
                    faulty_char = chr(((ord('A') + i + random.choice([-1, 1])) % 26) + 65)
                    labels.append(faulty_char)
                else:
                    labels.append(chr(65 + i))
        else:
            labels = []
        return labels
