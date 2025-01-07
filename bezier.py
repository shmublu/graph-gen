import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
from scipy.spatial.distance import cdist

class LetterGenerator:
    def __init__(self, canvas_size=1, line_probability=0.3, k_nearest=3, 
                 max_point_connections=3, max_direct_connections=1, 
                 discontinuity_probability=0.0):
        self.canvas_size = canvas_size
        self.line_probability = line_probability
        self.k_nearest = k_nearest
        self.max_point_connections = max_point_connections
        self.max_direct_connections = max_direct_connections
        self.discontinuity_probability = discontinuity_probability
        self.used_points = {}  # Track point usage count
        self.direct_connections = {}  # Track direct connections between points
        self.connected_points = set()  # Track which points have been connected
        
    def get_control_points(self, style='advanced_english'):
        control_points = {
            'digital_clock': [
                (0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0), (0.5, 1)
            ],
            'russian': [
                (0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0.5), (0.25, 0.75)
            ],
            'arabic': [
                (0, 0), (0.5, 0.5), (1, 0), (0.5, 1), (0.25, 0.25), (0.75, 0.75)
            ],
            'advanced_english': [
                (0, 0), (0.2, 0.8), (0.4, 0.6), (0.6, 0.4), (0.8, 0.2), (1, 0)
            ]
        }
        return control_points.get(style, control_points['advanced_english'])

    def get_k_nearest_points(self, current_point, all_points, k):
        if len(all_points) <= 1:
            return all_points
        
        points_array = np.array(all_points)
        current_array = np.array([current_point])
        distances = cdist(current_array, points_array)
        nearest_indices = np.argsort(distances[0])[:k]
        return [tuple(all_points[i]) for i in nearest_indices]

    def can_use_point(self, point):
        # Check if point has reached maximum connections
        if self.used_points.get(point, 0) >= self.max_point_connections:
            return False
        return True

    def can_connect_points(self, point1, point2):
        # Check direct connection limit
        connection = tuple(sorted([point1, point2]))
        if self.direct_connections.get(connection, 0) >= self.max_direct_connections:
            return False
        return True

    def update_point_usage(self, point1, point2):
        # Update point usage count
        self.used_points[point1] = self.used_points.get(point1, 0) + 1
        self.used_points[point2] = self.used_points.get(point2, 0) + 1
        
        # Update direct connection count
        connection = tuple(sorted([point1, point2]))
        self.direct_connections[connection] = self.direct_connections.get(connection, 0) + 1
        
        # Add points to connected set
        self.connected_points.add(point1)
        self.connected_points.add(point2)

    def get_discontinuity_point(self, points):
        unconnected_points = [p for p in points if p not in self.connected_points]
        connected_points = list(self.connected_points)
        
        if not unconnected_points or not connected_points:
            return None, None
            
        start_point = random.choice(unconnected_points)
        valid_end_points = [p for p in connected_points 
                          if self.can_use_point(p) and self.can_connect_points(start_point, p)]
        
        if not valid_end_points:
            return None, None
            
        end_point = random.choice(valid_end_points)
        return start_point, end_point

    def random_cubic_bezier(self, start_point, end_point, seed):
        random.seed(seed)
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        
        control_point1 = (
            start_point[0] + random.uniform(0.2, 0.8) * dx + random.uniform(-0.2, 0.2),
            start_point[1] + random.uniform(0.2, 0.8) * dy + random.uniform(-0.2, 0.2)
        )
        control_point2 = (
            start_point[0] + random.uniform(0.2, 0.8) * dx + random.uniform(-0.2, 0.2),
            start_point[1] + random.uniform(0.2, 0.8) * dy + random.uniform(-0.2, 0.2)
        )
        
        return [start_point, control_point1, control_point2, end_point]

    def generate_letter(self, style='advanced_english', num_curves=5, seed=42):
        random.seed(seed)
        self.used_points.clear()
        self.direct_connections.clear()
        self.connected_points.clear()
        points = self.get_control_points(style)
        curves = []
        
        current_point = random.choice(points)
        self.connected_points.add(current_point)
        
        for _ in range(num_curves):
            # Check for discontinuity
            if random.random() < self.discontinuity_probability:
                start_point, end_point = self.get_discontinuity_point(points)
                if start_point and end_point:
                    current_point = start_point
            
            available_points = [p for p in points 
                              if self.can_use_point(p) and 
                              p != current_point and 
                              self.can_connect_points(current_point, p)]
            
            if not available_points:
                break
                
            nearest_points = self.get_k_nearest_points(current_point, available_points, self.k_nearest)
            if not nearest_points:
                break
                
            end_point = random.choice(nearest_points)
            self.update_point_usage(current_point, end_point)
            
            is_line = random.random() < self.line_probability
            
            if is_line:
                curves.append([current_point, end_point])
            else:
                bezier_points = self.random_cubic_bezier(current_point, end_point, seed + _)
                curves.append(bezier_points)
            
            current_point = end_point
        
        return curves


curve_distribution = {
    3: 0.2,  # 20% chance of 3 curves
    4: 0.8,  # 80% chance of 4 curves
    5: 0.05  # 5% chance of 5 curves
}
def main1():
    # Normalize the distribution
    total = sum(curve_distribution.values())
    normalized_dist = {k: v/total for k, v in curve_distribution.items()}

    # Example usage to generate and display the alphabet
    fig, ax = plt.subplots(figsize=(15, 10))
    generator = LetterGenerator(
        canvas_size=1,
        line_probability=0.8,
        k_nearest=8,
        max_point_connections=1,
        max_direct_connections=1,
        discontinuity_probability=0.2
    )

    # Generate and display each letter
    for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        ax.clear()
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(f'Letter: {letter}')
        
        # Randomly select number of curves based on distribution
        num_curves = random.choices(list(normalized_dist.keys()), 
                                weights=list(normalized_dist.values()), 
                                k=1)[0]
        
        curves = generator.generate_letter('arabic', num_curves, i)
        
        for curve in curves:
            if len(curve) == 2:  # Line
                path = Path(curve, [Path.MOVETO, Path.LINETO])
            else:  # Bezier curve
                path = Path(curve, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
            patch = patches.PathPatch(path, facecolor='none', edgecolor='black', lw=2)
            ax.add_patch(patch)
        
        plt.pause(0.5)

    plt.show()