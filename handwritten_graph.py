import numpy as np
import random
import matplotlib.pyplot as plt
from graph_elements import BezierCurve, LineSegment
from handwritten import Handwriting


class HandwrittenGraph:
    """
    Base class for converting ideal graph shapes into a handwritten style.
    Includes common functionalities used by specific graph types.
    """

    def __init__(
        self,
        handwriting: Handwriting,
        config,
        debug=False
    ):
        """
        :param handwriting: an instance of Handwriting
        :param config: Configuration object containing graph options
        :param debug: If True, prints debug information
        """
        self.handwriting = handwriting
        self.config = config
        self.debug = debug
    def _define_char_map(self):
        """
        Define a mapping from characters to their corresponding curves.
        Each character is represented as a list of BezierCurve objects.
        Enhanced with more natural handwriting curves.
        """
        char_map = {}

        # Define digits 0-9 with natural curves
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
            '6': [
                BezierCurve([(0.8, 0.8), (0.6, 1), (0.4, 1), (0.2, 0.8)]),
                BezierCurve([(0.2, 0.8), (0, 0.6), (0, 0.4), (0.2, 0.2)]),
                BezierCurve([(0.2, 0.2), (0.4, 0), (0.6, 0), (0.8, 0.2)]),
                BezierCurve([(0.8, 0.2), (1, 0.4), (0.8, 0.6), (0.5, 0.5)]),
            ],
            '7': [
                BezierCurve([(0.1, 1), (0.3, 1), (0.7, 1), (0.9, 1)]),
                BezierCurve([(0.9, 1), (0.7, 0.7), (0.5, 0.3), (0.3, 0)]),
            ],
            '8': [
            BezierCurve([(0.5, 0.5), (0.3, 0.7), (0.7, 0.7), (0.5, 0.5)]),
            BezierCurve([(0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.5, 0.5)]),
            BezierCurve([(0.5, 0.5), (0.3, 0.7), (0.3, 0.3), (0.5, 0.5)]),
            BezierCurve([(0.5, 0.5), (0.7, 0.7), (0.7, 0.3), (0.5, 0.5)]),
        ],
            '9': [
                BezierCurve([(0.5, 0), (0.7, 0.2), (0.7, 0.4), (0.5, 0.5)]),
                BezierCurve([(0.5, 0.5), (0.3, 0.6), (0.3, 0.8), (0.5, 1)]),
                BezierCurve([(0.5, 1), (0.7, 1), (0.9, 0.8), (0.7, 0.6)]),
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
            ]
        })

        # Rework percent symbol
        char_map.update({
            '%': [
                BezierCurve([(0.1, 0.8), (0.2, 0.9), (0.3, 0.8), (0.2, 0.7)]),
                BezierCurve([(0.1, 0.2), (0.2, 0.3), (0.3, 0.2), (0.2, 0.1)]),
                BezierCurve([(0.1, 0.8), (0.8, 0.2)]),
            ],
        })

        # Add vowels, W, T
        char_map.update({
            'A': [
                BezierCurve([(0.1, 0), (0.3, 1), (0.5, 1), (0.7, 0)]),
                BezierCurve([(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5)]),
            ],
            'E': [
                BezierCurve([(0.8, 1), (0.2, 1), (0.2, 0.5)]),
                BezierCurve([(0.2, 0.5), (0.2, 0), (0.8, 0)]),
            ],
            'I': [
                BezierCurve([(0.4, 1), (0.6, 1), (0.6, 0)]),
                BezierCurve([(0.3, 0), (0.7, 0)]),
            ],
            'O': [
                BezierCurve([(0.2, 0.5), (0, 0.8), (0, 0.2), (0.2, 0.5)]),
                BezierCurve([(0.2, 0.5), (0.4, 0.8), (0.6, 0.8), (0.8, 0.5)]),
                BezierCurve([(0.8, 0.5), (0.6, 0.2), (0.4, 0.2), (0.2, 0.5)]),
            ],
            'U': [
                BezierCurve([(0, 0.8), (0, 0.2), (0.2, 0), (0.4, 0)]),
                BezierCurve([(0.4, 0), (0.6, 0), (0.8, 0.2), (0.8, 0.8)]),
            ],
            'W': [
                BezierCurve([(0, 0), (0.2, 1), (0.4, 0), (0.6, 1)]),
                BezierCurve([(0.6, 1), (0.8, 0), (1, 1), (1.2, 0)]),
            ],
            'T': [
                BezierCurve([(0, 1), (0.6, 1), (0.6, 1)]),
                BezierCurve([(0.3, 1), (0.3, 0)]),
            ],
            # Lowercase letters
            'a': [
                BezierCurve([(0.2, 0.5), (0.3, 0.7), (0.4, 0.7), (0.5, 0.5)]),
                BezierCurve([(0.5, 0.5), (0.6, 0.3), (0.7, 0.3), (0.8, 0.5)]),
                BezierCurve([(0.8, 0.5), (0.7, 0.4), (0.6, 0.4), (0.5, 0.5)]),
            ],
            'e': [
                BezierCurve([(0.5, 0.5), (0.2, 0.5), (0.2, 0.6), (0.5, 0.6)]),
                BezierCurve([(0.5, 0.6), (0.8, 0.6), (0.8, 0.4), (0.5, 0.4)]),
                BezierCurve([(0.5, 0.4), (0.5, 0.3), (0.5, 0.2), (0.5, 0.1)]),
            ],
            'i': [
                BezierCurve([(0.5, 0.8), (0.5, 0.9), (0.5, 1), (0.5, 1)]),
                BezierCurve([(0.5, 0.1), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)]),
            ],
            'o': [
                BezierCurve([(0.2, 0.5), (0, 0.8), (0, 0.2), (0.2, 0.5)]),
                BezierCurve([(0.2, 0.5), (0.4, 0.8), (0.6, 0.8), (0.8, 0.5)]),
                BezierCurve([(0.8, 0.5), (0.6, 0.2), (0.4, 0.2), (0.2, 0.5)]),
            ],
            'u': [
                BezierCurve([(0, 0.8), (0, 0.2), (0.2, 0), (0.4, 0)]),
                BezierCurve([(0.4, 0), (0.6, 0), (0.8, 0.2), (0.8, 0.8)]),
            ],
            'w': [
                BezierCurve([(0, 0), (0.2, 1), (0.4, 0), (0.6, 1)]),
                BezierCurve([(0.6, 1), (0.8, 0), (1, 1), (1.2, 0)]),
            ],
            't': [
                BezierCurve([(0.5, 1), (0.5, 1)]),
                BezierCurve([(0.3, 1), (0.3, 0.5), (0.3, 0), (0.3, 0)]),
            ],
        })

        # Add lowercase letters if space permits
        char_map.update({
            'a': [
                BezierCurve([(0.2, 0.5), (0.3, 0.7), (0.4, 0.7), (0.5, 0.5)]),
                BezierCurve([(0.5, 0.5), (0.6, 0.3), (0.7, 0.3), (0.8, 0.5)]),
                BezierCurve([(0.8, 0.5), (0.7, 0.4), (0.6, 0.4), (0.5, 0.5)]),
            ],
            'e': [
                BezierCurve([(0.5, 0.5), (0.2, 0.5), (0.2, 0.6), (0.5, 0.6)]),
                BezierCurve([(0.5, 0.6), (0.8, 0.6), (0.8, 0.4), (0.5, 0.4)]),
                BezierCurve([(0.5, 0.4), (0.5, 0.3), (0.5, 0.2), (0.5, 0.1)]),
            ],
            'i': [
                BezierCurve([(0.5, 0.8), (0.5, 0.9), (0.5, 1), (0.5, 1)]),
                BezierCurve([(0.5, 0.1), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)]),
            ],
            'o': [
                BezierCurve([(0.2, 0.5), (0, 0.8), (0, 0.2), (0.2, 0.5)]),
                BezierCurve([(0.2, 0.5), (0.4, 0.8), (0.6, 0.8), (0.8, 0.5)]),
                BezierCurve([(0.8, 0.5), (0.6, 0.2), (0.4, 0.2), (0.2, 0.5)]),
            ],
            'u': [
                BezierCurve([(0, 0.8), (0, 0.2), (0.2, 0), (0.4, 0)]),
                BezierCurve([(0.4, 0), (0.6, 0), (0.8, 0.2), (0.8, 0.8)]),
            ],
            'w': [
                BezierCurve([(0, 0), (0.2, 1), (0.4, 0), (0.6, 1)]),
                BezierCurve([(0.6, 1), (0.8, 0), (1, 1), (1.2, 0)]),
            ],
            't': [
                BezierCurve([(0.5, 1), (0.5, 1)]),
                BezierCurve([(0.3, 1), (0.3, 0.5), (0.3, 0), (0.3, 0)]),
            ],
        })

        return char_map
    def _add_noise(self, value, noise_level):
        """Add noise to a value based on the specified noise level."""
        return value + random.uniform(-noise_level, noise_level)

    def render_text(self, text, position, scale=0.5, accuracy=1.0):
        print(text)
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


    def draw_graph(
        self,
        data,
        labels=None
    ):
        """
        Draw the specified graph with all options in a single call.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    ########################################################################
    # Common Drawing Methods
    ########################################################################
    def _apply_handwriting_to_text(self, text_segments, scale_factor):
        """
        Apply handwriting effects to text segments after scaling.

        :param text_segments: List of BezierCurve or LineSegment objects representing text
        :param scale_factor: Scaling factor for text
        :return: List of handwritten curves
        """
        scaled_segments = []

        # Scale the text segments
        for segment in text_segments:
            if isinstance(segment, LineSegment):
                scaled_start = segment.start_point * scale_factor
                scaled_end = segment.end_point * scale_factor
                scaled_segments.append(LineSegment(scaled_start, scaled_end))
            elif isinstance(segment, BezierCurve):
                scaled_control_points = [point * scale_factor for point in segment.control_points]
                scaled_segments.append(BezierCurve(scaled_control_points))

        # Discretize and apply handwriting effects with adjusted parameters for text
        point_curves = []
        for segment in scaled_segments:
            discrete_points = [segment.evaluate(t) for t in np.linspace(0, 1, 50)]
            point_curves.append(discrete_points)

        handwritten_curves = self.handwriting.apply_handwriting(point_curves)

        return handwritten_curves

    def draw_axes(self, x_range, y_range, labels=None):
        """
        Draw X and Y axes with optional ticks and labels.
        """
        if self.debug:
            print(f"[DEBUG] draw_axes called with x_range={x_range}, y_range={y_range}")

        angle_rad = np.deg2rad(self.config.axis_angle_degrees)

        def rotate_point(px, py):
            """Rotate (px, py) around the origin by angle_rad."""
            rx = px * np.cos(angle_rad) - py * np.sin(angle_rad)
            ry = px * np.sin(angle_rad) + py * np.cos(angle_rad)
            return (rx, ry)

        curves = []

        # X Axis
        x_start = rotate_point(x_range[0], 0)
        x_end = rotate_point(x_range[1], 0)
        curves.append(LineSegment(x_start, x_end))

        # Y Axis
        y_start = rotate_point(0, y_range[0])
        y_end = rotate_point(0, y_range[1])
        curves.append(LineSegment(y_start, y_end))

        # Add ticks
        if self.config.show_ticks:
            tick_interval = self.config.tick_interval
            for x in range(int(np.floor(x_range[0])), int(np.floor(x_range[1])) + 1, tick_interval):
                if not self.config.number_axes and labels:
                    continue
                if random.random() < self.config.tick_skip_prob:
                    continue
                tick_length = self.config.tick_length
                tick_bottom = rotate_point(x, -tick_length)
                tick_top = rotate_point(x, tick_length)
                curves.append(LineSegment(tick_bottom, tick_top))

            for y in range(int(np.floor(y_range[0])), int(np.floor(y_range[1])) + 1, tick_interval):
                if not self.config.number_axes and labels:
                    continue
                if random.random() < self.config.tick_skip_prob:
                    continue
                tick_length = self.config.tick_length
                tick_left = rotate_point(-tick_length, y)
                tick_right = rotate_point(tick_length, y)
                curves.append(LineSegment(tick_left, tick_right))

        # Discretize and apply handwriting
        point_curves = []
        for segment in curves:
            discrete_points = [segment.evaluate(t) for t in np.linspace(0, 1, 50)]
            point_curves.append(discrete_points)

        handwritten_curves = self.handwriting.apply_handwriting(point_curves)

        # Add axis labels if numbered_axes
        if self.config.number_axes and labels:
            label_segments = []
            # X Axis Label
            x_label = labels[0]
            x_label_position = ((x_range[0] + x_range[1]) / 2, -0.8) if self.config.center_axis_labels else (x_range[1] + 0.5, -0.8)
            label_segments += self.render_text(x_label, x_label_position, accuracy=self.config.label_accuracy)

            # Y Axis Label
            y_label = labels[1]
            y_label_position = (-0.8, (y_range[0] + y_range[1]) / 2) if self.config.center_axis_labels else (-0.8, y_range[1] + 0.5)
            label_segments += self.render_text(y_label, y_label_position, accuracy=self.config.label_accuracy)

            # Convert label segments to point curves
            for seg in label_segments:
                discrete_points = [seg.evaluate(t) for t in np.linspace(0, 1, 50)]
                handwritten_label = self.handwriting.apply_handwriting([discrete_points])
                handwritten_curves += handwritten_label

        return handwritten_curves
