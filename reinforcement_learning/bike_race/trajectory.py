from math import cos, sin, tan, atan2, radians, pi, degrees, sqrt
import pygame
import numpy as np

# cost weights
BOUNDS_WEIGHT = 10
COLLISION_WEIGHT = 100
DISTANCE_WEIGHT = -1/1000

RELATIVE_PROGRESS_WEIGHT = 1/5
PROXIMITY_WEIGHT = 1
OVERLAP_WEIGHT = 1

GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 130, 80)

def bounding_box(points):
    """
    Compute the bounding box of a set of points.

    Args:
        points (list of tuples): List of points (x, y).

    Returns:
        tuple: (min_x, min_y, max_x, max_y) defining the bounding box.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def boxes_intersect(box1, box2):
    """
    Check if two bounding boxes intersect.

    Args:
        box1 (tuple): (min_x, min_y, max_x, max_y) for box 1.
        box2 (tuple): (min_x, min_y, max_x, max_y) for box 2.

    Returns:
        bool: True if boxes intersect, False otherwise.
    """
    return not (box1[2] < box2[0] or box1[0] > box2[2] or
                box1[3] < box2[1] or box1[1] > box2[3])


def intersecting_area(box1, box2):
    """
    Calculate the area of intersection between two bounding boxes.

    Args:
        box1 (tuple): (min_x, min_y, max_x, max_y) for box 1.
        box2 (tuple): (min_x, min_y, max_x, max_y) for box 2.

    Returns:
        float: Area of the intersection, or 0 if the boxes do not intersect.
    """
    # Calculate the intersection bounds
    inter_min_x = max(box1[0], box2[0])
    inter_min_y = max(box1[1], box2[1])
    inter_max_x = min(box1[2], box2[2])
    inter_max_y = min(box1[3], box2[3])

    # Compute the width and height of the intersection
    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)

    # Return the intersection area
    return inter_width * inter_height


def intersect(line1, line2):
    """
    Checks if two line segments intersect.

    Args:
        line1 (list): A list of two tuples representing the endpoints of the first line segment.
        line2 (list): A list of two tuples representing the endpoints of the second line segment.

    Returns:
        bool: True if the line segments intersect, False otherwise.
    """

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = line1
    C, D = line2

    is_same_point = A == C or B == D or A == D or B == C
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D) or is_same_point


class Trajectory:
    def __init__(self,  course, bike, color, number=0):

        self.min_x = 10000
        self.max_x = -10000
        self.min_y = 10000
        self.max_y = -10000

        self.points = []
        self.total_absolute_cost = 0
        self.collision_cost = 0
        self.bounds_cost = 0
        self.distance_cost = 0
        self.collision_weight = COLLISION_WEIGHT

        self.color = color
        self.course = course
        self.bike = bike
        self.length = 0

        self.is_displaying = False
        self.is_chosen = False
        self.is_collision_checked = False
        self.number = number

        self.intersecting_trajectories = []
        self.trajectory_proximity_costs =  np.zeros((len(self.bike.action_lst) ** self.bike.mpc_horizon))
        self.trajectory_overlap_costs =  np.zeros((len(self.bike.action_lst) ** self.bike.mpc_horizon))
        self.relative_arc_length_costs =  np.zeros((len(self.bike.action_lst) ** self.bike.mpc_horizon))
        self.total_relative_costs = np.zeros((len(self.bike.action_lst) ** self.bike.mpc_horizon))
        self.point_count = 0


    def draw(self, screen):
        """
          Draw the trajectory and its associated cost on the screen.
          The index is used to space out cost text to avoid overlapping.
          """
        for pt in self.points:
            pygame.draw.circle(screen, self.color, (pt[0], pt[1]), 1)

        # Draw costs near trajectories with spacing adjustment
        if self.is_displaying:
            font = pygame.font.Font(None, 20)
            cost_text = font.render(f"{round(self.total_absolute_cost):.0f}", True, BLACK)
            num_text = font.render(f"{self.number}", True, BLACK)
            text_x = self.points[-1][0] + 10
            text_y = self.points[-1][1] - 10

            screen.blit(num_text, (text_x, text_y))

    def update(self):
        for other_traj in self.intersecting_trajectories:
            if other_traj.is_chosen:
                self.collision_cost += COLLISION_WEIGHT
                self.total_absolute_cost = self.bounds_cost + self.distance_cost + self.collision_cost
                self.total_relative_cost = self.relative_arc_length_costs + self.trajectory_proximity_costs + self.trajectory_overlap_costs

    def add_point(self, x, y):

        # update min/max of trajectory for bounding boxes
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

        self.bounds_cost += BOUNDS_WEIGHT * self.check_bounds(x, y)
        if self.bounds_cost > 0:
            self.color = RED

        # collision_cost = COLLISION_WEIGHT * self.check_collision(x, y)

        self.length = self.calc_arc_length_distance(x, y)
        self.distance_cost += DISTANCE_WEIGHT * self.length

        # self.total_cost += round(bounds_cost + collision_cost + self.distance_cost, 2)
        # collision cost added in bike class as result of interactions
        self.total_absolute_cost = round(self.bounds_cost + self.distance_cost, 2)

        self.points.append((round(x, 2), round(y, 2)))

    def get_bounding_box(self):
        """Retrieve bounding box in O(1)."""
        return (self.min_x, self.min_y, self.max_x,  self.max_y)

    def check_bounds(self, new_x, new_y):
        # Calculate the distance from the center to the point
        distance_squared = (new_x - self.course.center_x) ** 2 + (new_y - self.course.center_y) ** 2

        # Check if the distance is between the outer radius and inner radius
        if self.course.inner_radius ** 2 <= distance_squared <= self.course.outer_radius ** 2:
            return 0
        else:
            return 1

    def arc_length(self, x, y):
        # Calculate angular position in radians
        theta = atan2(y - self.course.center_y, x - self.course.center_x)
        theta = (theta + 2 * pi) % (2 * pi)  # Normalize to [0, 2π)

        # Compute arc length
        arc_length = theta * self.course.outer_radius
        return arc_length

    def calc_arc_length_distance(self, x, y):
        # Calculate arc lengths for both points
        arc1 = self.arc_length(self.bike.x, self.bike.y)
        arc2 = self.arc_length(x, y)

        # Calculate the absolute distance, handling wraparound
        distance = abs(arc2 - arc1)
        if distance > pi * self.course.outer_radius:  # Adjust for crossing the start/finish line
            distance = 2 * pi * self.course.outer_radius - distance
        return distance

    def trajectory_intersection(self, other_traj):
        for (pt1, pt2) in zip(self.points[:-2], self.points[1:]):
            for (pt3, pt4) in zip(other_traj.points[:-2], other_traj.points[1:]):

                if intersect([pt1, pt2], [pt3, pt4]):
                    self.intersecting_trajectories.append(other_traj)


    def trajectory_sensing(self, other_traj, action_interval, mpc_horizon):
        """
        Check if two trajectories intersect using bounding box filtering.

        Args:
            other_traj (Trajectory): Another trajectory to check intersection with.

        Returns:
            bool: True if the trajectories intersect, False otherwise.
        """
        # relative arc length
        other_end_pos = other_traj.points[-1]
        end_pos = self.points[-1]

        arc_length = self.arc_length(end_pos[0], end_pos[1])
        other_arc_length = self.arc_length(other_end_pos[0], other_end_pos[1])

        # negative is good, incentive
        relative_arc_length = other_arc_length - arc_length
        self.relative_arc_length_costs[other_traj.number] = relative_arc_length * RELATIVE_PROGRESS_WEIGHT
        other_traj.relative_arc_length_costs[self.number] = -relative_arc_length * RELATIVE_PROGRESS_WEIGHT

        # proximity
        distance = abs(sqrt((end_pos[0]-other_end_pos[0])**2+(end_pos[1]-other_end_pos[1])**2))
        self.trajectory_proximity_costs[other_traj.number] = distance * PROXIMITY_WEIGHT
        other_traj.trajectory_proximity_costs[self.number] = distance * PROXIMITY_WEIGHT

        # overlap
        # Compute bounding boxes
        box1 = self.get_bounding_box()
        box2 = other_traj.get_bounding_box()

        # If bounding boxes don't overlap, trajectories don't intersect
        if boxes_intersect(box1, box2):
            area = intersecting_area(box1, box2)
            self.trajectory_overlap_costs[other_traj.number] = area * OVERLAP_WEIGHT
            other_traj.trajectory_overlap_costs[self.number] = area * OVERLAP_WEIGHT

            # length must be multiple of action interval size
            length_interval = action_interval * mpc_horizon
            (pt1, pt2) = self.points[0], self.points[length_interval - 1]
            (pt3, pt4) = other_traj.points[0], other_traj.points[length_interval - 1]

            if intersect([pt1, pt2], [pt3, pt4]):
                self.intersecting_trajectories.append(other_traj)
                other_traj.intersecting_trajectories.append(self)
                self.color = ORANGE
                other_traj.color = ORANGE
                return True
        return False
