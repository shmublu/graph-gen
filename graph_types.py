import numpy as np
import random
import matplotlib.pyplot as plt
from graph_elements import BezierCurve, LineSegment
from handwritten_graph import HandwrittenGraph


class BarGraph(HandwrittenGraph):
    """
    Class for drawing handwritten bar graphs.
    """

    def draw_graph(self, data, labels=None):
        """
        Draw the bar graph with the given data and labels.
        """
        if self.debug:
            print("[DEBUG] Drawing Bar Graph.")
        self.char_map = self._define_char_map()
        bar_positions = data['bar_positions']
        bar_heights = data['bar_heights']
        curves = []

        for idx, (pos, height) in enumerate(zip(bar_positions, bar_heights)):
            pos = self._add_noise(pos, self.config.spacing_noise) if not self.config.align_bars else pos
            height = self._add_noise(height, self.config.scale_noise)

            bar_width = self.config.bar_width
            bottom_left = (pos - bar_width, 0)
            top_left = (pos - bar_width, height)
            top_right = (pos + bar_width, height)
            bottom_right = (pos + bar_width, 0)

            curves.extend([
                LineSegment(bottom_left, top_left),
                LineSegment(top_left, top_right),
                LineSegment(top_right, bottom_right),
                LineSegment(bottom_right, bottom_left)
            ])

            # Add labels on top of bars
            if labels and len(labels) > idx:
                label = labels[idx]
                label_position = (pos, height + self.config.label_offset)
                label_segments = self.render_text(label, label_position, accuracy=self.config.label_accuracy)
                curves.extend(label_segments)

        # Discretize and apply handwriting
        point_curves = []
        for segment in curves:
            if isinstance(segment, LineSegment) or isinstance(segment, BezierCurve):
                discrete_points = [segment.evaluate(t) for t in np.linspace(0, 1, 50)]
                point_curves.append(discrete_points)
            else:
                for seg in segment:
                    discrete_points = [seg.evaluate(t) for t in np.linspace(0, 1, 50)]
                    point_curves.append(discrete_points)

        handwritten_curves = self.handwriting.apply_handwriting(point_curves)

        # Draw axes
        axes_curves = self.draw_axes(
            x_range=self.config.x_range,
            y_range=self.config.y_range,
            labels=self.config.axis_labels
        )
        handwritten_curves += axes_curves

        # Plot all curves
        self._plot_curves(handwritten_curves, self.config.title)

    def _plot_curves(self, curves, title):
        """
        Plot all handwritten curves.
        """
        plt.figure(figsize=(self.config.width, self.config.height))

        for stroke in curves:
            xvals, yvals = zip(*stroke)
            plt.plot(xvals, yvals, color='black', linewidth=2)

        # Set plot properties
        plt.title(title, fontsize=14)
        plt.axis('equal')
        plt.axis('off')  # Hide default axes

        # Apply log scale if needed
        if self.config.log_scale:
            if self.config.graph_type in ['bar', 'scatter']:
                plt.yscale("log")
            if self.config.graph_type == 'scatter':
                plt.xscale("log")

        plt.show()


class ScatterPlot(HandwrittenGraph):
    """
    Class for drawing handwritten scatter plots.
    """

    def draw_graph(self, data, labels=None):
        """
        Draw the scatter plot with the given data and labels.
        """
        if self.debug:
            print("[DEBUG] Drawing Scatter Plot.")
        self.char_map = self._define_char_map()
        x_values = data['scatter_x']
        y_values = data['scatter_y']
        curves = []

        for idx, (x, y) in enumerate(zip(x_values, y_values)):
            marker_type = self.config.markers
            if marker_type in ['circle', 'o']:
                radius = self.config.marker_size
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
                    discrete_circle.extend([seg.evaluate(t) for t in np.linspace(0, 1, 50)])
                curves.append(discrete_circle)
            elif marker_type == 'x':
                size = self.config.marker_size * 2
                cpoints = [
                    (x - size, y - size),
                    (x + size, y + size),
                    (x - size, y + size),
                    (x + size, y - size)
                ]
                for i in range(len(cpoints) - 1):
                    seg = LineSegment(cpoints[i], cpoints[i + 1])
                    curves.append([seg.evaluate(t) for t in np.linspace(0, 1, 50)])

            # Add labels based on probability
            if labels and len(labels) > idx and random.random() < self.config.scatter_label_probability:
                label = labels[idx]
                label_position = (x, y + self.config.scatter_label_offset)
                label_segments = self.render_text(label, label_position, accuracy=self.config.label_accuracy)
                curves.extend(label_segments)

        # Best-fit line
        if self.config.scatter_show_best_fit and len(x_values) > 1:
            relevant_points = [(xx, yy) for xx, yy in zip(x_values, y_values) if not self.config.log_scale or (xx > 0 and yy > 0)]
            if len(relevant_points) > 1:
                xs, ys = zip(*relevant_points)
                slope, intercept = np.polyfit(xs, ys, 1)
                x_min, x_max = min(xs), max(xs)
                best_fit_line = LineSegment(
                    (x_min, slope * x_min + intercept),
                    (x_max, slope * x_max + intercept)
                )
                bf_points = [best_fit_line.evaluate(t) for t in np.linspace(0, 1, 100)]
                curves.append(bf_points)

        # Discretize and apply handwriting
        point_curves = []
        for curve in curves:
            if isinstance(curve, list):
                point_curves.append(curve)
            elif isinstance(curve, (BezierCurve, LineSegment)):
                discrete_points = [curve.evaluate(t) for t in np.linspace(0, 1, 50)]
                point_curves.append(discrete_points)

        handwritten_curves = self.handwriting.apply_handwriting(point_curves)

        # Draw axes
        axes_curves = self.draw_axes(
            x_range=self.config.x_range,
            y_range=self.config.y_range,
            labels=self.config.axis_labels
        )
        handwritten_curves += axes_curves

        # Plot all curves
        self._plot_curves(handwritten_curves, self.config.title)

    def _plot_curves(self, curves, title):
        """
        Plot all handwritten curves.
        """
        plt.figure(figsize=(self.config.width, self.config.height))

        for stroke in curves:
            xvals, yvals = zip(*stroke)
            plt.plot(xvals, yvals, color='black', linewidth=2)

        # Set plot properties
        plt.title(title, fontsize=14)
        plt.axis('equal')
        plt.axis('off')  # Hide default axes

        # Apply log scale if needed
        if self.config.log_scale:
            if self.config.graph_type in ['bar', 'scatter']:
                plt.yscale("log")
            if self.config.graph_type == 'scatter':
                plt.xscale("log")

        plt.show()


class PieChart(HandwrittenGraph):
    """
    Class for drawing handwritten pie charts.
    """

    def draw_graph(self, data, labels=None):
        """
        Draw the pie chart with the given data and labels.
        """
        if self.debug:
            print("[DEBUG] Drawing Pie Chart.")
        self.char_map = self._define_char_map()  # Ensure character map is initialized
        slices = data['pie_slices']
        radius = self.config.radius
        curves = []
        total = sum(slices)
        angles = [2 * np.pi * (s / total) for s in slices]
        current_angle = 0.0

        # Draw slices
        for idx, angle in enumerate(angles):
            end_angle = current_angle + angle
            num_points = 50

            # Create an arc for the slice
            arc = [
                (radius * np.cos(theta), radius * np.sin(theta))
                for theta in np.linspace(current_angle, end_angle, num_points)
            ]
            curves.append(arc)

            # Add a line from the center to the end of the arc
            curves.append([(0, 0), (radius * np.cos(current_angle), radius * np.sin(current_angle))])
            curves.append([(0, 0), (radius * np.cos(end_angle), radius * np.sin(end_angle))])

            current_angle = end_angle

            # Add labels
            if labels and len(labels) > idx:
                mid_angle = current_angle - angle / 2
                label_position = (
                    (1 + self.config.pie_label_distance) * radius * np.cos(mid_angle),
                    (1 + self.config.pie_label_distance) * radius * np.sin(mid_angle)
                )
                label_segments = self.render_text(labels[idx], label_position, accuracy=self.config.label_accuracy)

                # Apply handwriting effect to text after scaling
                if label_segments:
                    handwritten_text_curves = self._apply_handwriting_to_text(label_segments, scale_factor=1.5)
                    curves.extend(handwritten_text_curves)

        # Discretize and apply handwriting to pie chart elements
        point_curves = []
        for curve in curves:
            if isinstance(curve, list):  # Directly append lists of points
                point_curves.append(curve)
            elif isinstance(curve, (BezierCurve, LineSegment)):  # Discretize Bezier or LineSegment objects
                discrete_points = [curve.evaluate(t) for t in np.linspace(0, 1, 50)]
                point_curves.append(discrete_points)

        handwritten_curves = self.handwriting.apply_handwriting(point_curves)

        # Plot all curves
        self._plot_curves(handwritten_curves, self.config.title)

    def _plot_curves(self, curves, title):
        """
        Plot all handwritten curves.
        """
        plt.figure(figsize=(self.config.width, self.config.height))

        for stroke in curves:
            xvals, yvals = zip(*stroke)
            plt.plot(xvals, yvals, color='black', linewidth=2)

        # Set plot properties
        plt.title(title, fontsize=14)
        plt.axis('equal')
        plt.axis('off')  # Hide default axes

        # Apply log scale if needed
        if self.config.log_scale:
            plt.yscale("log")
            plt.xscale("log")

        plt.show()
