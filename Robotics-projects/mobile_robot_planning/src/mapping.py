#!/usr/bin/env python3

"""
    # {student full name}
    # {student id}
    # {student email}
"""

# Python standard library
from math import cos, sin, atan2, fabs

# Numpy
import numpy as np

# "Local version" of ROS messages
from local.geometry_msgs import PoseStamped, Quaternion
from local.sensor_msgs import LaserScan
from local.nav_msgs import OccupancyGrid
from local.map_msgs import OccupancyGridUpdate

from grid_map import GridMap


class Mapping:
    def __init__(self, unknown_space, free_space, c_space, occupied_space,
                 radius, optional=None):
        self.unknown_space = unknown_space
        self.free_space = free_space
        self.c_space = c_space
        self.occupied_space = occupied_space
        self.allowed_values_in_map = {"self.unknown_space": self.unknown_space,
                                      "self.free_space": self.free_space,
                                      "self.c_space": self.c_space,
                                      "self.occupied_space": self.occupied_space}
        self.radius = radius
        self.__optional = optional

    def get_yaw(self, q):
        """Returns the Euler yaw from a quaternion.
        :type q: Quaternion
        """
        return atan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))

    def raytrace(self, start, end):
        """Returns all cells in the grid map that has been traversed
        from start to end, including start and excluding end.
        start = (x, y) grid map index
        end = (x, y) grid map index
        """
        (start_x, start_y) = start
        (end_x, end_y) = end
        x = start_x
        y = start_y
        (dx, dy) = (fabs(end_x - start_x), fabs(end_y - start_y))
        n = dx + dy
        x_inc = 1
        if end_x <= start_x:
            x_inc = -1
        y_inc = 1
        if end_y <= start_y:
            y_inc = -1
        error = dx - dy
        dx *= 2
        dy *= 2

        traversed = []
        for i in range(0, int(n)):
            traversed.append((int(x), int(y)))

            if error > 0:
                x += x_inc
                error -= dy
            else:
                if error == 0:
                    traversed.append((int(x + x_inc), int(y)))
                y += y_inc
                error += dx

        return traversed

    def add_to_map(self, grid_map, x, y, value):
        """Adds value to index (x, y) in grid_map if index is in bounds.
        Returns weather (x, y) is inside grid_map or not.
        """
        if value not in self.allowed_values_in_map.values():
            raise Exception("{0} is not an allowed value to be added to the map. "
                            .format(value) + "Allowed values are: {0}. "
                            .format(self.allowed_values_in_map.keys()) +
                            "Which can be found in the '__init__' function.")

        if self.is_in_bounds(grid_map, x, y):
            grid_map[x, y] = value
            return True
        return False

    def is_in_bounds(self, grid_map, x, y):
        """Returns weather (x, y) is inside grid_map or not."""
        if x >= 0 and x < grid_map.get_width():
            if y >= 0 and y < grid_map.get_height():
                return True
        return False
    
    def update_map(self, grid_map, pose, scan):

        """Updates the grid_map with the data from the laser scan and the pose.
        """
        robot_yaw = self.get_yaw(pose.pose.orientation)
        origine = grid_map.get_origin()
        risoluz = grid_map.get_resolution()

        """
        Fill in your solution here
        """

        pos_robot_x = pose.pose.position.x
        pos_robot_y = pose.pose.position.y   #pos robot in m
        origine_mappa_x = origine.position.x
        origine_mappa_y = origine.position.y

        robot_indice_x = int((pos_robot_x - origine_mappa_x) / risoluz)
        robot_indice_y = int((pos_robot_y - origine_mappa_y) / risoluz)
        robot_pos_grid = (robot_indice_x, robot_indice_y)

        mappa_aggiornata = False
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
 
        # liste in cui se salvano i punti
        obstacle_points = []
        free_cells = [] 

        for indice_dist, distanza in enumerate(scan.ranges):
            if distanza <= scan.range_min or distanza >= scan.range_max:
                continue

            angolo_distanza = scan.angle_min + (indice_dist * scan.angle_increment)

            ostacolo_x_rispetto_al_robot = distanza * cos(angolo_distanza)
            ostacolo_y_rispetto_al_robot = distanza * sin(angolo_distanza)

            ostacolox_ruotato = ostacolo_x_rispetto_al_robot * cos(robot_yaw) - ostacolo_y_rispetto_al_robot * sin(robot_yaw)
            ostacoloy_ruotato = ostacolo_x_rispetto_al_robot * sin(robot_yaw) + ostacolo_y_rispetto_al_robot * cos(robot_yaw)

            ostacolox_nel_mondo = ostacolox_ruotato + pos_robot_x
            ostacoloy_nel_mondo = ostacoloy_ruotato + pos_robot_y

            indice_x_map = int((ostacolox_nel_mondo - origine_mappa_x) / risoluz)
            indice_y_map = int((ostacoloy_nel_mondo - origine_mappa_y) / risoluz)

            end_point_grid = (indice_x_map, indice_y_map)
            obstacle_points.append(end_point_grid)
            
            # aggiung le celle libere alla lista
            traversed_cells = self.raytrace(robot_pos_grid, end_point_grid)
            free_cells.extend(traversed_cells) 
        unique_free_cells = list(set(free_cells)) 
        
        for (x, y) in unique_free_cells:
            # Sovrascrive solo se è sconosciuto
            if self.is_in_bounds(grid_map, x, y):
                self.add_to_map(grid_map, x, y, self.free_space)
                mappa_aggiornata = True
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

        # aggiorna la mappa
        #sovrascrive le celle libere
        for (x, y) in obstacle_points:
            if self.add_to_map(grid_map, x, y, self.occupied_space):
                mappa_aggiornata = True
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        """
        For C only!
        Fill in the update correctly below.
        """
        update = OccupancyGridUpdate()

        if mappa_aggiornata:
            update.x = min_x
            update.y = min_y
            update.width = (max_x - min_x) + 1
            update.height = (max_y - min_y) + 1

            update.data = []
            for y_row in range(update.y, update.y + update.height):
                for x_col in range(update.x, update.x + update.width):
                    update.data.append(grid_map[x_col, y_row])
        else:
            update.x = 0
            update.y = 0
            update.width = 0
            update.height = 0
            update.data = []

        return grid_map, update

    def inflate_map(self, grid_map):
        
        """For C only!
        Inflate the map with self.c_space assuming the robot
        has a radius of self.radius.
        
        Returns the inflated grid_map.

        Inflating the grid_map means that for each self.occupied_space
        you calculate and fill in self.c_space. Make sure to not overwrite
        something that you do not want to.


        You should use:
            self.c_space  # For C space (inflated space).
            self.radius   # To know how much to inflate.

            You can use the function add_to_map to be sure that you add
            values correctly to the map.

            You can use the function is_in_bounds to check if a coordinate
            is inside the map.

        :type grid_map: GridMap
        """


        """
        Fill in your solution here
        """

        width = grid_map.get_width()
        height = grid_map.get_height()

        # trova tutti gli ost
        occupied_cells = []
        for x in range(width):
            for y in range(height):
                if grid_map[x, y] == self.occupied_space:
                    occupied_cells.append((x, y))

        r = int(self.radius)
        r_squared = r * r

        # per ogni ostacolo trovato
        for (cx, cy) in occupied_cells:
            for x in range(cx - r, cx + r + 1):
                for y in range(cy - r, cy + r + 1):
                    if self.is_in_bounds(grid_map, x, y):
                        if ((x - cx)**2 + (y - cy)**2) <= r_squared:

                            if grid_map[x, y] != self.occupied_space:
                                self.add_to_map(grid_map, x, y, self.c_space)
        return grid_map
