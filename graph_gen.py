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


##########################################################
# HandwrittenGraph Class
##########################################################
class HandwrittenGraph:
    """
    Converts ideal graph shapes (bars, scatter points, pie arcs, axes, etc.)
    into a handwritten style by creating control points or line segments and
    passing them through the Handwriting class.
    """

    def __init__(
        self,
        handwriting,
        non_idealness=0.1,
        spacing_noise=0.1,
        scale_noise=0.1,
        tick_noise=0.05,
        debug=False,
        show_ticks=True,
        axis_angle_degrees=0.0,
        show_best_fit=False,
        log_scale=False,
        align_bars=True,
        label_type='number',
        label_options=None,
        axis_labels=('X', 'Y'),
        center_axis_labels=False
    ):
        """
        :param handwriting: an instance of Handwriting
        :param non_idealness: overall control for random variations
        :param spacing_noise: controls noise in spacing between elements
        :param scale_noise: controls noise in scaling of elements
        :param tick_noise: controls noise in tick sizes and labels
        :param debug: if True, prints debug info
        :param show_ticks: if True, axes will have ticks
        :param axis_angle_degrees: angle offset for the entire axes
        :param show_best_fit: if True, a best-fit line is drawn in scatterplot
        :param log_scale: if True, use log scale for the y-axis (in bar/scatter)
        :param align_bars: if True, align bars with axes by default
        :param label_type: 'number' or 'alphabet' for labeling
        :param label_options: dictionary of labeling options
        :param axis_labels: tuple containing labels for x and y axes
        :param center_axis_labels: if True, center the axis labels
        """
        self.handwriting = handwriting
        self.non_idealness = non_idealness
        self.spacing_noise = spacing_noise
        self.scale_noise = scale_noise
        self.tick_noise = tick_noise
        self.debug = debug
        self.show_ticks = show_ticks
        self.axis_angle_degrees = axis_angle_degrees
        self.show_best_fit = show_best_fit
        self.log_scale = log_scale
        self.align_bars = align_bars
        self.label_type = label_type
        self.label_options = label_options or {}
        self.axis_labels = axis_labels
        self.center_axis_labels = center_axis_labels

        # Define simple character curves for digits and some letters
        self.char_map = self._define_char_map()

    def _define_char_map(self):
        """
        Define a mapping from characters to their corresponding curves.
        Each character is represented as a list of BezierCurve objects.
        Enhanced with more natural handwriting curves.
        """
        char_map = {}

        # Define digits 0-9 and letters with natural curves
        char_map.update({
            '0': [
                BezierCurve([(0.2, 0.8), (0, 0.6), (0, 0.4), (0.2, 0.2)]),
                BezierCurve([(0.2, 0.2), (0.4, 0), (0.6, 0), (0.8, 0.2)]),
                BezierCurve([(0.8, 0.2), (1, 0.4), (1, 0.6), (0.8, 0.8)]),
                BezierCurve([(0.8, 0.8), (0.6, 1), (0.4, 1), (0.2, 0.8)]),
            ],
            '1': [
                BezierCurve([(0.3, 0.8), (0.4, 0.9), (0.5, 1), (0.5, 1)]),
                BezierCurve([(0.5, 1), (0.5, 0.8), (0.5, 0.2), (0.5, 0)]),
                BezierCurve([(0.3, 0), (0.4, 0), (0.6, 0), (0.7, 0)]),
            ],
            '2': [
                BezierCurve([(0.2, 0.8), (0.3, 1), (0.7, 1), (0.8, 0.8)]),
                BezierCurve([(0.8, 0.8), (0.9, 0.6), (0.7, 0.4), (0.4, 0.2)]),
                BezierCurve([(0.4, 0.2), (0.2, 0.1), (0.2, 0), (0.8, 0)]),
            ],
            '3': [
                BezierCurve([(0.2, 1), (0.4, 1.1), (0.7, 1), (0.8, 0.8)]),
                BezierCurve([(0.8, 0.8), (0.9, 0.6), (0.7, 0.5), (0.5, 0.5)]),
                BezierCurve([(0.5, 0.5), (0.7, 0.5), (0.9, 0.4), (0.8, 0.2)]),
                BezierCurve([(0.8, 0.2), (0.7, 0), (0.3, 0), (0.2, 0.1)]),
            ],
            '4': [
                BezierCurve([(0.8, 0), (0.8, 0.3), (0.8, 0.7), (0.8, 1)]),
                BezierCurve([(0.2, 0.8), (0.4, 0.8), (0.6, 0.4), (0.8, 0.4)]),
                BezierCurve([(0.8, 0.4), (0.6, 0.4), (0.4, 0.4), (0.2, 0.4)]),
            ],
            '5': [
                BezierCurve([(0.8, 1), (0.4, 1), (0.2, 1), (0.2, 0.7)]),
                BezierCurve([(0.2, 0.7), (0.3, 0.6), (0.6, 0.6), (0.8, 0.5)]),
                BezierCurve([(0.8, 0.5), (1, 0.3), (0.8, 0.1), (0.5, 0)]),
                BezierCurve([(0.5, 0), (0.3, 0), (0.1, 0.1), (0.2, 0.2)]),
            ],
            'A': [
                BezierCurve([(0, 0), (0.2, 0.3), (0.4, 0.7), (0.5, 1)]),
                BezierCurve([(0.5, 1), (0.6, 0.7), (0.8, 0.3), (1, 0)]),
                BezierCurve([(0.2, 0.4), (0.4, 0.4), (0.6, 0.4), (0.8, 0.4)]),
            ],
            'B': [
                BezierCurve([(0.2, 0), (0.2, 0.3), (0.2, 0.7), (0.2, 1)]),
                BezierCurve([(0.2, 1), (0.5, 1), (0.8, 0.9), (0.8, 0.7)]),
                BezierCurve([(0.8, 0.7), (0.8, 0.6), (0.6, 0.5), (0.2, 0.5)]),
                BezierCurve([(0.2, 0.5), (0.6, 0.5), (0.9, 0.4), (0.9, 0.2)]),
                BezierCurve([(0.9, 0.2), (0.9, 0), (0.5, 0), (0.2, 0)]),
            ],
            'C': [
                BezierCurve([(0.8, 0.8), (0.6, 1), (0.4, 1), (0.2, 0.8)]),
                BezierCurve([(0.2, 0.8), (0, 0.6), (0, 0.4), (0.2, 0.2)]),
                BezierCurve([(0.2, 0.2), (0.4, 0), (0.6, 0), (0.8, 0.2)]),
            ],
            'D': [
                BezierCurve([(0.2, 0), (0.2, 0.3), (0.2, 0.7), (0.2, 1)]),
                BezierCurve([(0.2, 1), (0.5, 1), (0.8, 0.8), (0.9, 0.5)]),
                BezierCurve([(0.9, 0.5), (0.8, 0.2), (0.5, 0), (0.2, 0)]),
            ],
            'E': [
                BezierCurve([(0.8, 1), (0.5, 1), (0.2, 1), (0.2, 0.8)]),
                BezierCurve([(0.2, 0.8), (0.2, 0.6), (0.2, 0.4), (0.2, 0.2)]),
                BezierCurve([(0.2, 0.2), (0.2, 0), (0.5, 0), (0.8, 0)]),
                BezierCurve([(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.7, 0.5)]),
            ],
            'F': [
                BezierCurve([(0.8, 1), (0.5, 1), (0.2, 1), (0.2, 0.8)]),
                BezierCurve([(0.2, 0.8), (0.2, 0.6), (0.2, 0.2), (0.2, 0)]),
                BezierCurve([(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.7, 0.5)]),
            ],
            'X': [
                BezierCurve([(0.1, 1), (0.3, 0.7), (0.7, 0.3), (0.9, 0)]),
                BezierCurve([(0.1, 0), (0.3, 0.3), (0.7, 0.7), (0.9, 1)]),
            ],
            'Y': [
                BezierCurve([(0.1, 1), (0.3, 0.7), (0.5, 0.5), (0.5, 0.5)]),
                BezierCurve([(0.9, 1), (0.7, 0.7), (0.5, 0.5), (0.5, 0.5)]),
                BezierCurve([(0.5, 0.5), (0.5, 0.3), (0.5, 0.1), (0.5, 0)]),
            ],
            '%': [
                BezierCurve([(0.1, 0.8), (0.2, 0.9), (0.3, 0.8), (0.2, 0.7)]),
                BezierCurve([(0.1, 0.2), (0.2, 0.3), (0.3, 0.2), (0.2, 0.1)]),
                BezierCurve([(0.1, 0.8), (0.8, 0.2)])
            ],
            '.': [
                BezierCurve([(0.3, 0.1), (0.3, 0.1), (0.4, 0.1), (0.4, 0.1)])
            ],
        })

        return char_map

    def _add_noise(self, value, noise_level):
        """Add noise to a value based on the specified noise level."""
        return value + random.uniform(-noise_level, noise_level)

    def render_text(self, text, position, scale=0.5, accuracy=1.0):
        """Render text by converting each character into curves."""
        text = str(text).upper()  # Ensure text is string and uppercase
        x_offset, y_offset = position
        spacing = 0.6 * scale
        curves = []

        # Handle decimal numbers properly
        if '.' in text:
            integer_part, decimal_part = text.split('.')
            text = integer_part + '.' + decimal_part[:1]  # Keep only one decimal place

        for char in text:
            # Skip error introduction for decimal points and other symbols
            if random.random() > accuracy and char.isdigit():
                faulty_char = str((int(char) + random.choice([-1, 1])) % 10)
                segments = self.char_map.get(faulty_char, [])
            else:
                segments = self.char_map.get(char, [])
                if not segments and char == '.':
                    # Add simple dot representation if missing
                    segments = [BezierCurve([(0.3, 0.1), (0.3, 0.1), (0.4, 0.1), (0.4, 0.1)])]

            # Scale and position the segments
            for segment in segments:
                if isinstance(segment, (LineSegment, BezierCurve)):
                    if isinstance(segment, LineSegment):
                        scaled_start = (
                            segment.start_point[0] * scale + x_offset,
                            segment.start_point[1] * scale + y_offset
                        )
                        scaled_end = (
                            segment.end_point[0] * scale + x_offset,
                            segment.end_point[1] * scale + y_offset
                        )
                        curves.append(LineSegment(scaled_start, scaled_end))
                    else:
                        scaled_control = [
                            (pt[0] * scale + x_offset, pt[1] * scale + y_offset)
                            for pt in segment.control_points
                        ]
                        curves.append(BezierCurve(scaled_control))
            x_offset += spacing

        return curves

    ########################################################################
    # Axis
    ########################################################################
    def draw_axes(self, x_range, y_range, labels=None, tick_interval=1, tick_scale=1.0, tick_skip_prob=0.0, label_accuracy=1.0):
        """
        Draw X and Y axes with optional ticks and labels.
        By default, from x_range[0] to x_range[1] and y_range[0] to y_range[1].
        Axis can be angled by self.axis_angle_degrees if desired.
        :param tick_interval: Interval between tick labels.
        :param tick_scale: Scale multiplier for tick labels.
        :param tick_skip_prob: Probability to skip a tick label.
        :param label_accuracy: Probability that axis labels are rendered correctly.
        """
        if self.debug:
            print(f"[DEBUG] draw_axes called with x_range={x_range}, y_range={y_range}")
            print(f"         axis_angle_degrees={self.axis_angle_degrees}, show_ticks={self.show_ticks}")

        # Convert angle to radians
        angle_rad = np.deg2rad(self.axis_angle_degrees)

        def rotate_point(px, py):
            """Rotate (px, py) around the origin by angle_rad."""
            rx = px * np.cos(angle_rad) - py * np.sin(angle_rad)
            ry = px * np.sin(angle_rad) + py * np.cos(angle_rad)
            return (rx, ry)

        # -----------------------------
        # Create axes as LineSegments
        # -----------------------------
        curves = []
        
        # X Axis: from (x_range[0], 0) to (x_range[1], 0)
        x_start = rotate_point(x_range[0], 0)
        x_end = rotate_point(x_range[1], 0)
        curves.append(LineSegment(x_start, x_end))

        # Y Axis: from (0, y_range[0]) to (0, y_range[1])
        y_start = rotate_point(0, y_range[0])
        y_end = rotate_point(0, y_range[1])
        curves.append(LineSegment(y_start, y_end))

        # -----------------------------
        # Add ticks if needed
        # -----------------------------
        if self.show_ticks:
            start_xi = int(np.floor(x_range[0]))
            end_xi = int(np.floor(x_range[1]))
            for x in range(start_xi, end_xi + 1, int(tick_interval)):
                if random.random() < tick_skip_prob:
                    continue  # Skip this tick label based on probability
                tick_length_bot = -0.2 + self._add_noise(-0.2, self.tick_noise)
                tick_length_top = 0.2 + self._add_noise(0.2, self.tick_noise)
                tick_bottom = rotate_point(x, tick_length_bot)
                tick_top = rotate_point(x, tick_length_top)
                curves.append(LineSegment(tick_bottom, tick_top))

            start_yi = int(np.floor(y_range[0]))
            end_yi = int(np.floor(y_range[1]))
            for y in range(start_yi, end_yi + 1, int(tick_interval)):
                if random.random() < tick_skip_prob:
                    continue  # Skip this tick label based on probability
                tick_length_left = -0.2 + self._add_noise(-0.2, self.tick_noise)
                tick_length_right = 0.2 + self._add_noise(0.2, self.tick_noise)
                tick_left = rotate_point(tick_length_left, y)
                tick_right = rotate_point(tick_length_right, y)
                curves.append(LineSegment(tick_left, tick_right))

        # -----------------------------
        # Discretize the axis lines
        # -----------------------------
        point_curves = []
        for segment in curves:
            discrete_points = []
            if isinstance(segment, LineSegment):
                discrete_points = [
                    segment.evaluate(t) for t in np.linspace(0, 1, 50)
                ]
            elif isinstance(segment, BezierCurve):
                discrete_points = [
                    segment.evaluate(t) for t in np.linspace(0, 1, 50)
                ]
            point_curves.append(discrete_points)

        handwritten_curves = self.handwriting.apply_handwriting(point_curves)

        # -----------------------------
        # Add axis labels
        # -----------------------------
        if labels:
            label_line_segments = []
            # X Axis Label
            x_label = labels[0]
            if self.center_axis_labels:
                x_label_position = (
                    (x_range[0] + x_range[1]) / 2,
                    -0.8  # Slightly below the x-axis
                )
            else:
                x_label_position = (x_range[1] + 0.5, -0.8)
            label_segments_x = self.render_text(x_label, x_label_position, accuracy=label_accuracy)
            label_line_segments.extend(label_segments_x)

            # Y Axis Label
            y_label = labels[1]
            if self.center_axis_labels:
                y_label_position = (
                    -0.8, 
                    (y_range[0] + y_range[1]) / 2
                )
            else:
                y_label_position = (-0.8, y_range[1] + 0.5)
            label_segments_y = self.render_text(y_label, y_label_position, accuracy=label_accuracy)
            label_line_segments.extend(label_segments_y)

            # Convert label line segments to point lists
            label_point_curves = []
            for seg in label_line_segments:
                if isinstance(seg, LineSegment) or isinstance(seg, BezierCurve):
                    discrete_points = [seg.evaluate(t) for t in np.linspace(0, 1, 50)]
                    label_point_curves.append(discrete_points)

            # Pass the label points to handwriting
            handwritten_labels = self.handwriting.apply_handwriting(label_point_curves)
            handwritten_curves += handwritten_labels

        return handwritten_curves

    ########################################################################
    # Bar Graph
    ########################################################################
    def draw_bar_graph(self, bar_positions, bar_heights, labels=None, label_accuracy=1.0):
        """
        Convert bar graph data into line segments (each bar's boundary) and apply handwriting.
        Optionally align bars with axes.
        Optionally add labels to bars.
        :param label_accuracy: Probability that each bar label is rendered correctly.
        """
        if self.debug:
            print("[DEBUG] draw_bar_graph called.")
            print(f"         bar_positions={bar_positions}")
            print(f"         bar_heights={bar_heights}")

        curves = []
        for idx, (pos, height) in enumerate(zip(bar_positions, bar_heights)):
            # Optionally add a small random offset to pos if not aligning
            pos = self._add_noise(pos, self.spacing_noise) if not self.align_bars else pos
            height = self._add_noise(height, self.scale_noise)

            bar_width = 0.4
            bottom_left = (pos - bar_width, 0)
            top_left = (pos - bar_width, height)
            top_right = (pos + bar_width, height)
            bottom_right = (pos + bar_width, 0)

            curves.append(LineSegment(bottom_left, top_left))
            curves.append(LineSegment(top_left, top_right))
            curves.append(LineSegment(top_right, bottom_right))
            curves.append(LineSegment(bottom_right, bottom_left))

            # Add labels on top of bars
            if labels and len(labels) > idx:
                label = labels[idx]
                label_position = (pos, height + 0.5)
                label_segments = self.render_text(label, label_position, accuracy=label_accuracy)
                curves.extend(label_segments)

        # Convert each segment to an array of points (or use them directly if already lists)
        point_curves = []
        for segment in curves:
            if isinstance(segment, LineSegment) or isinstance(segment, BezierCurve):
                discrete_points = [
                    segment.evaluate(t) for t in np.linspace(0, 1, 50)
                ]
                point_curves.append(discrete_points)
            else:
                # Assume it's already a list of segments (from render_text)
                for seg in segment:
                    if isinstance(seg, LineSegment) or isinstance(seg, BezierCurve):
                        discrete_points = [
                            seg.evaluate(t) for t in np.linspace(0, 1, 50)
                        ]
                        point_curves.append(discrete_points)

        handwritten_curves = self.handwriting.apply_handwriting(point_curves)
        return handwritten_curves

    ########################################################################
    # Scatter Plot
    ########################################################################
    def draw_scatterplot(self, x_values, y_values, labels=None, markers='o', label_probability=1.0, label_accuracy=1.0):
        """
        Convert scatterplot data into markers and apply handwriting.
        Optionally add labels to points.
        :param label_probability: Probability that each point is labeled.
        :param label_accuracy: Probability that each label is rendered correctly.
        """
        import numpy as np  # Ensure numpy is available

        if self.debug:
            print("[DEBUG] draw_scatterplot called.")
            print(f"         x_values={x_values}")
            print(f"         y_values={y_values}")
            print(f"         show_best_fit={self.show_best_fit}")
            print(f"         log_scale={self.log_scale}")
            print(f"         markers={markers}")

        # Create a set of curves for each point based on marker type
        point_curves = []
        for idx, (x, y) in enumerate(zip(x_values, y_values)):
            if markers in ['circle', 'o']:
                radius = 0.1
                cpoints = [
                    (x + radius, y),
                    (x, y + radius),
                    (x - radius, y),
                    (x, y - radius),
                    (x + radius, y)
                ]
                discrete_circle = []
                for i in range(len(cpoints) - 1):
                    seg = LineSegment(cpoints[i], cpoints[i + 1])
                    seg_points = [seg.evaluate(t) for t in np.linspace(0, 1, 50)]
                    discrete_circle.extend(seg_points)
                point_curves.append(discrete_circle)
            elif markers == 'x':
                size = 0.2
                cpoints = [
                    (x - size, y - size),
                    (x + size, y + size),
                    (x - size, y + size),
                    (x + size, y - size)
                ]
                for i in range(len(cpoints) - 1):
                    seg = LineSegment(cpoints[i], cpoints[i + 1])
                    seg_points = [seg.evaluate(t) for t in np.linspace(0, 1, 50)]
                    point_curves.append(seg_points)

            # Add labels to scatter points based on probability
            if labels and len(labels) > idx and random.random() < label_probability:
                label = labels[idx]
                label_position = (x, y + 0.3)
                label_segments = self.render_text(label, label_position, accuracy=label_accuracy)
                for seg in label_segments:
                    if isinstance(seg, LineSegment) or isinstance(seg, BezierCurve):
                        seg_points = [seg.evaluate(t) for t in np.linspace(0, 1, 50)]
                        point_curves.append(seg_points)

        # If best-fit line is enabled, compute a linear regression
        best_fit_curves = []
        if self.show_best_fit and len(x_values) > 1:
            # Filter out negative or zero if log scale is used (simple approach)
            if self.log_scale:
                relevant_points = [
                    (xx, yy) for xx, yy in zip(x_values, y_values)
                    if xx > 0 and yy > 0
                ]
            else:
                relevant_points = list(zip(x_values, y_values))

            if len(relevant_points) > 1:
                xs, ys = zip(*relevant_points)
                # simple linear best-fit
                slope, intercept = np.polyfit(xs, ys, 1)
                x_min, x_max = min(xs), max(xs)
                best_fit_line = LineSegment(
                    (x_min, slope * x_min + intercept),
                    (x_max, slope * x_max + intercept)
                )
                bf_points = [best_fit_line.evaluate(t) for t in np.linspace(0, 1, 100)]
                best_fit_curves.append(bf_points)

        # Combine scatter markers + optional best-fit line
        all_curves = point_curves + best_fit_curves
        handwritten_curves = self.handwriting.apply_handwriting(all_curves)
        return handwritten_curves

    ########################################################################
    # Pie Chart
    ########################################################################
    def draw_pie_chart(self, slices, radius=1.0, labels=None, label_accuracy=1.0):
        """
        Draw a simple pie chart with arcs; each slice is an arc of the circle,
        plus two short radius lines to enclose the slice. Optionally add labels.
        :param slices: list of slice fractions that sum to 1.0
        :param radius: radius of the pie chart
        :param labels: list of labels for each slice
        :param label_accuracy: Probability that each label is rendered correctly.
        """


        if self.debug:
            print(f"[DEBUG] draw_pie_chart called with slices={slices}, radius={radius}")

        curves = []
        
        # Draw the complete circle first as one continuous curve
        npoints = 100
        circle_points = []
        for i in range(npoints + 1):
            alpha = 2 * pi * i / npoints
            r = radius + self._add_noise(radius, self.scale_noise) * 0.02
            circle_points.append((r * np.cos(alpha), r * np.sin(alpha)))
        curves.append(circle_points)

        # Add radius lines
        current_angle = 0.0
        for idx, fraction in enumerate(slices):
            theta = current_angle
            current_angle += 2 * pi * fraction
            
            # Single radius line for each division
            radius_line = [
                (0, 0),
                (radius * np.cos(theta), radius * np.sin(theta))
            ]
            curves.append(radius_line)

            # Add labels
            if labels and len(labels) > idx:
                label = labels[idx]
                mid_angle = theta + (pi * fraction)
                label_position = (
                    1.2 * radius * np.cos(mid_angle),
                    1.2 * radius * np.sin(mid_angle)
                )
                label_segments = self.render_text(label, label_position, accuracy=label_accuracy)
                curves.extend(label_segments)

        # Convert label curves (if any)
        point_curves = []
        for segment in curves:
            if isinstance(segment, list):
                if all(isinstance(pt, tuple) for pt in segment):
                    point_curves.append(segment)
                else:
                    for seg in segment:
                        if isinstance(seg, LineSegment) or isinstance(seg, BezierCurve):
                            discrete_points = [
                                seg.evaluate(t) for t in np.linspace(0, 1, 50)
                            ]
                            point_curves.append(discrete_points)
            elif isinstance(segment, LineSegment) or isinstance(segment, BezierCurve):
                discrete_points = [
                    segment.evaluate(t) for t in np.linspace(0, 1, 50)
                ]
                point_curves.append(discrete_points)

        handwritten_curves = self.handwriting.apply_handwriting(point_curves)
        return handwritten_curves


##########################################################
# Main Execution: Show separate graphs
##########################################################

def main():
    # Initialize GraphGenerator with desired non-idealness
    graph_gen = GraphGenerator(spacing_noise=0.2, scale_noise=0.2, tick_noise=0.05)

    # 1) Generate random data
    bar_positions, bar_heights = graph_gen.generate_bar_graph_data(num_bars=7)
    scatter_x, scatter_y = graph_gen.generate_scatterplot_data(num_points=15)
    pie_slices = graph_gen.generate_pie_data(num_slices=5)

    # Generate labels with error_rate for bar and scatter labels
    bar_labels = [f"{height:.1f}" for height in bar_heights]
    scatter_labels = graph_gen.generate_labels(label_type='number', count=15, error_rate=0.1)  # 10% error
    pie_labels = [f"{int(s*100)}%" for s in pie_slices]

    # 2) Initialize Handwriting with desired parameters
    handwriting = Handwriting(
        base_speed=1.0,
        mass=1.0,
        damping=0.9,
        min_duration=0.2,
        steps_per_unit=50,
        max_force=-15.0,
        max_force_change=-1,
        debug=False,   # Set to True if you want verbose debugging from handwriting
        verify=True,
        tremor_channels=2,
        tremor_params=[
            {'max': 100, 'prob': 0.9, 'max_length': 100},
            {'max': 1000, 'prob': 0.03, 'max_length': 210}
        ]
    )

    # 3) Initialize HandwrittenGraph with various toggles
    #    Try flipping these to see different behaviors
    handwritten_graph_bar = HandwrittenGraph(
        handwriting=handwriting,
        non_idealness=0.2,
        spacing_noise=0.2,
        scale_noise=0.2,
        tick_noise=0.05,
        debug=False,
        show_ticks=True,          # Whether to show ticks in the axes
        axis_angle_degrees=0.0,   # Slight slant
        show_best_fit=False,      # Not relevant in bar chart
        log_scale=False,          # If True, would need positive data for logs
        align_bars=True,          # Align bars with axes by default
        label_type='number',      # Use number labels
        label_options={'position': 'top'},
        axis_labels=('Age', 'Median Salary (USD)'),
        center_axis_labels=True
    )

    handwritten_graph_scatter = HandwrittenGraph(
        handwriting=handwriting,
        non_idealness=0.0,
        spacing_noise=0.2,
        scale_noise=0.2,
        tick_noise=0.05,
        debug=False,
        show_ticks=True,
        axis_angle_degrees=0.0,
        show_best_fit=True,       # We add a best-fit line in scatter
        log_scale=False,          # If True, would apply log scale
        align_bars=False,
        label_type='number',
        label_options={'position': 'right'},
        axis_labels=('Ages', 'Median Salary (USD)'),
        center_axis_labels=True
    )

    handwritten_graph_pie = HandwrittenGraph(
        handwriting=handwriting,
        non_idealness=0.2,
        spacing_noise=0.1,
        scale_noise=0.2,
        tick_noise=0.0,
        debug=False,
        show_ticks=False,         # Not actually used in pie chart
        axis_angle_degrees=0.0,
        show_best_fit=False,
        log_scale=False,
        align_bars=False,
        label_type='number',
        label_options={'position': 'outside'},
        axis_labels=('Category', ''),
        center_axis_labels=True
    )

    ########################################################################
    # BAR GRAPH
    ########################################################################
    plt.figure(figsize=(7, 5))
    axes_curves = handwritten_graph_bar.draw_axes(
        x_range=(0, 8),
        y_range=(0, 12),
        labels=handwritten_graph_bar.axis_labels,
        tick_interval=1,
        tick_scale=1.0,
        tick_skip_prob=0.1,        # 10% chance to skip a tick label
        label_accuracy=0.9          # 90% accuracy for axis labels
    )
    bar_curves = handwritten_graph_bar.draw_bar_graph(bar_positions, bar_heights, labels=bar_labels, label_accuracy=0.9)
    all_curves_bar = axes_curves + bar_curves

    if handwritten_graph_bar.log_scale:
        # If you want to do log scale for the bar graph:
        plt.yscale("log")

    for stroke in all_curves_bar:
        xvals, yvals = zip(*stroke)
        plt.plot(xvals, yvals, color='black', linewidth=2)

    plt.title("Handwritten Bar Graph", fontsize=14)
    plt.axis('equal')
    plt.axis('off')  # Hide axes lines beyond our hand-drawn ones
    plt.show()

    ########################################################################
    # SCATTER PLOT
    ########################################################################
    plt.figure(figsize=(7, 5))
    axes_curves_scat = handwritten_graph_scatter.draw_axes(
        x_range=(-1, 11),
        y_range=(-1, 11),
        labels=handwritten_graph_scatter.axis_labels,
        tick_interval=2,
        tick_scale=1.0,
        tick_skip_prob=0.1,        # 10% chance to skip a tick label
        label_accuracy=0.9          # 90% accuracy for axis labels
    )
    scatter_curves = handwritten_graph_scatter.draw_scatterplot(
        scatter_x, 
        scatter_y, 
        labels=scatter_labels, 
        markers='circle',
        label_probability=0.8,      # 80% chance to label each point
        label_accuracy=0.9           # 90% accuracy for point labels
    )
    all_curves_scatter = axes_curves_scat + scatter_curves

    if handwritten_graph_scatter.log_scale:
        plt.yscale("log")
        plt.xscale("log")

    for stroke in all_curves_scatter:
        xvals, yvals = zip(*stroke)
        plt.plot(xvals, yvals, color='black', linewidth=2)

    plt.title("Handwritten Scatter Plot", fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

    ########################################################################
    # PIE CHART
    ########################################################################
    plt.figure(figsize=(6, 6))
    pie_curves = handwritten_graph_pie.draw_pie_chart(
        slices=pie_slices, 
        radius=2.0, 
        labels=pie_labels,
        label_accuracy=0.9
    )
    for stroke in pie_curves:
        xvals, yvals = zip(*stroke)
        plt.plot(xvals, yvals, color='black', linewidth=2)

    plt.title("Handwritten Pie Chart", fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
