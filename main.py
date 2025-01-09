from graph_elements import GraphGenerator
from handwritten_graph import HandwrittenGraph
from graph_types import BarGraph, ScatterPlot, PieChart
from handwritten import Handwriting
import random


class GraphConfig:
    """
    Configuration class for all graph types.
    Centralizes all options for easy management.
    """

    def __init__(self):
        # Common settings
        self.width = 15
        self.height = 5
        self.non_idealness = 0.1
        self.spacing_noise = 0.1
        self.scale_noise = 0.1
        self.tick_noise = 0.05
        self.debug = False
        self.show_ticks = True
        self.tick_length = 0.1
        self.number_axes = True
        self.axis_angle_degrees = 0.0
        self.log_scale = False
        self.center_axis_labels = True
        self.label_accuracy = 1.0
        self.axis_labels = ('X', 'Y')
        self.show_best_fit = False
        self.align_bars = True
        self.label_type = 'number'

        # Bar Graph specific
        self.bar_width = 0.4
        self.bar_label_offset = 0.5

        # Scatter Plot specific
        self.marker_size = 0.1
        self.scatter_label_probability = 0.8
        self.scatter_label_offset = 0.3
        self.scatter_show_best_fit = True
        self.markers = 'circle'

        # Pie Chart specific
        self.radius = 1.0
        self.pie_label_distance = 0.2
        self.show_axes_pie = False

        # Titles
        self.bar_title = "Handwritten Bar Graph"
        self.scatter_title = "Handwritten Scatter Plot"
        self.pie_title = "Handwritten Pie Chart"


def main():
    # Initialize configuration
    config = GraphConfig()

    # Initialize GraphGenerator with desired non-idealness
    graph_gen = GraphGenerator(
        spacing_noise=config.spacing_noise,
        scale_noise=config.scale_noise,
        tick_noise=config.tick_noise
    )

    # Generate data
    bar_positions, bar_heights = graph_gen.generate_bar_graph_data(num_bars=7)
    scatter_x, scatter_y = graph_gen.generate_scatterplot_data(num_points=15)
    pie_slices = graph_gen.generate_pie_data(num_slices=5)

    # Generate labels with error_rate for bar and scatter labels
    bar_labels = [f"{height:.1f}" for height in bar_heights]
    scatter_labels = graph_gen.generate_labels(label_type='number', count=15, error_rate=0.1)  # 10% error
    pie_labels = [f"{int(s*100)}%" for s in pie_slices]

    # Initialize Handwriting with desired parameters
    handwriting = Handwriting(
        base_speed=1.0,
        mass=1.0,
        damping=0.6,
        min_duration=0.2,
        steps_per_unit=50,
        max_force=-15.0,
        max_force_change=-1,
        debug=False,   # Set to True for verbose debugging
        verify=True,
        tremor_channels=2,
        tremor_params=[
            {'max': 120, 'prob': 0.3, 'max_length': 40},
            {'max': 1150, 'prob': 0.03, 'max_length': 250}
        ]
    )

    # Initialize and draw Bar Graph
    bar_config = config
    bar_config.x_range = (0, 8)
    bar_config.y_range = (0, 12)
    bar_config.tick_interval = 1
    bar_config.tick_scale = 1.0
    bar_config.tick_skip_prob = 0.0
    bar_config.title = bar_config.bar_title
    bar_config.align_bars = True
    bar_config.label_accuracy = 1.0
    bar_config.bar_width = 0.4
    bar_config.label_offset = bar_config.bar_label_offset

    handwritten_graph_bar = BarGraph(
        handwriting=handwriting,
        config=bar_config,
        debug=config.debug
    )
    handwritten_graph_bar.draw_graph(
        data={'bar_positions': bar_positions, 'bar_heights': bar_heights},
        labels=bar_labels
    )

    # Initialize and draw Scatter Plot
    scatter_config = config
    scatter_config.x_range = (-1, 11)
    scatter_config.y_range = (-1, 11)
    scatter_config.tick_interval = 2
    scatter_config.tick_scale = 1.0
    scatter_config.tick_skip_prob = 0.1
    scatter_config.title = scatter_config.scatter_title
    scatter_config.show_best_fit = True
    scatter_config.log_scale = False
    scatter_config.align_bars = False
    scatter_config.label_type = 'number'
    scatter_config.label_accuracy = 0.9
    scatter_config.scatter_label_probability = 0.8
    scatter_config.marker_type = 'circle'
    scatter_config.label_offset = scatter_config.scatter_label_offset

    handwritten_graph_scatter = ScatterPlot(
        handwriting=handwriting,
        config=scatter_config,
        debug=config.debug
    )
    handwritten_graph_scatter.draw_graph(
        data={'scatter_x': scatter_x, 'scatter_y': scatter_y},
        labels=scatter_labels
    )

    # Initialize and draw Pie Chart
    pie_config = config
    pie_config.radius = 2.0
    pie_config.label_accuracy = 0.9
    pie_config.title = pie_config.pie_title
    pie_config.label_distance = pie_config.pie_label_distance
    pie_config.show_axes = pie_config.show_axes_pie

    handwritten_graph_pie = PieChart(
        handwriting=handwriting,
        config=pie_config,
        debug=config.debug
    )
    handwritten_graph_pie.draw_graph(
        data={'pie_slices': pie_slices},
        labels=pie_labels
    )


if __name__ == "__main__":
    main()
