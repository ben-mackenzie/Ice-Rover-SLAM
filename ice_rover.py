"""
 === Introduction ===

   A few months ago a new rover was sent to McMurdo Station in the Antarctic. The rover is a technical marvel
   as it is equipped with the latest scientific sensors and analyzers capable of surviving the harsh climate of the
   South Pole.  The goal is for the rover to reach a series of test sites and perform scientific sampling and analysis.
   Due to the extreme conditions, the rover will be air dropped via parachute into the test area.  The good news is
   the surface is smooth and free from any type of obstacles, the bad news is the surface is entirely ice which may
   introduce noise into your rovers movements.  The station scientists are ready to deploy the new rover, but first
   we need to create and test the planning software that will be used on board to ensure it can complete it's goals.

   The assignment is broken up into two parts.

   Part A:
        Create a SLAM implementation to process a series of landmark (beacon) measurements and movement updates.

        Hint: A planner with an unknown number of motions works well with an online version of SLAM.

    Part B:
        Here you will create the planner for the rover.  The rover does unfortunately has a series of limitations:

        - Start position
          - The rover will land somewhere within range of at least 3 or more beacon for measurements.

        - Measurements
          - Measurements will come from beacons and test sites within range of the rover's antenna horizon.
            * The format is {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'beacon'}, ...}
          - Satellites and test sites will always return a measurement if in range.

        - Movements
          - Action: 'move 1.570963 1.0'
            * The rover will turn counterclockwise 90 degrees and then move 1.0
          - stochastic due to the icy surface.
          - if max distance or steering is exceeded, the rover will not move.

        - Samples
          - Provided as list of x and y coordinates, [[0., 0.], [1., -3.5], ...]
          - Action: 'sample'
            * The rover will attempt to take a sample at the current location.
          - A rover can only take a sample once per requested site.
          - The rover must be with 0.25 distance to successfully take a sample.
            * Hint: Be sure to account for floating point limitations
          - The is a 100ms penalty if the robot is requested to sample a site not on the list or if the site has
            previously been sampled.
          - Use sys.stdout = open('stdout.txt', 'w') to directly print data if necessary.

        The rover will always execute a measurement first, followed by an action.

        The rover will have a time limit of 5 seconds to find and sample all required sites.
"""

from matrix import *
from robot import *

class SLAM:
    """Create a basic SLAM module.
    """
    def __init__(self):
        """Initialize SLAM components here.
        """
        self.bearing = 0
        self.landmark_IDs = {}
        self.Omega = matrix()
        self.Xi = matrix()
        self.Omega.identity(2)
        self.Xi.zero(2,1)

    def process_measurements(self, measurements):
        """Process a new series of measurements.

        Args:
            measurements(dict): Collection of measurements
                in the format {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'beacon'}, ...}

        Returns:
            x, y: current belief in location of the rover relative to initial location before movement
        """

        for id, input_dict in measurements.iteritems():
            distance = input_dict['distance']
            bearing = input_dict['bearing']
            m_type = input_dict['type']
            adjusted_bearing = truncate_angle(self.bearing + bearing)

            # calculate dx, dy from bearing and distance
            dx = distance * cos(adjusted_bearing)
            dy = distance * sin(adjusted_bearing)
            distance_update = [dx, dy]

            # calculate measurement noise
            distance_sigma = 0.05 * distance
            bearing_sigma = 0.02 * distance
            bearing_noise = random.gauss(0, bearing_sigma)
            sigma_x = sqrt((distance_sigma ** 2) * (cos(adjusted_bearing) ** 2) + (distance ** 2) * (bearing_sigma ** 2) * (sin(adjusted_bearing) ** 2))
            sigma_y = sqrt((distance_sigma ** 2) * (sin(adjusted_bearing) ** 2) + (distance ** 2) * (bearing_sigma ** 2) * (cos(adjusted_bearing) ** 2))
            x_noise = abs(random.gauss(0, sigma_x))*50
            y_noise = abs(random.gauss(0, sigma_y))*50
            measurement_noise = [x_noise, y_noise]

            # If seeing landmark for the first time, add to matrices and landmark tracking dict
            if id not in self.landmark_IDs:
                # expand matrices
                rows = len(self.Omega.value)
                columns = len(self.Omega.value[0])
                self.Omega = self.Omega.expand(rows + 2, columns + 2, [r for r in range(rows)], [c for c in range(columns)])
                self.Xi = self.Xi.expand(rows + 2, 1, [r for r in range(rows)], [0])
                # add landmark ID and corresponding index to landmark tracking dictionary
                self.landmark_IDs[id] = (len(self.landmark_IDs) + 1) * 2

            # update matrix values
            m = self.landmark_IDs[id]
            for b in range(2):
                self.Omega.value[b][b] += 1.0 / measurement_noise[b]
                self.Omega.value[m + b][m + b] += 1.0 / measurement_noise[b]
                self.Omega.value[b][m + b] += -1.0 / measurement_noise[b]
                self.Omega.value[m + b][b] += -1.0 / measurement_noise[b]
                self.Xi.value[b][0] += -distance_update[b] / measurement_noise[b]
                self.Xi.value[m + b][0] += distance_update[b] / measurement_noise[b]

        # calculating mu
        mu = self.Omega.inverse() * self.Xi
        x, y = mu.value[0][0], mu.value[1][0]

        return x, y

    def process_movement(self, steering, distance, motion_noise=0.01):
        """Process a new movement.

        Args:
            steering(float): amount to turn
            distance(float): distance to move
            motion_noise(float): movement noise

        Returns:
            x, y: current belief in location of the rover relative to initial location after movement
        """
        #update robot bearing
        self.bearing += steering

        # calculate dx, dy from bearing and distance
        dx = distance * cos(self.bearing)
        dy = distance * sin(self.bearing)
        distance_update = [dx, dy]

        # expand Omega and Xi to accommodate new motion
        rows = len(self.Omega.value)
        columns = len(self.Omega.value[0])
        indices_of_original_elements = [0,1] + [r for r in range(4,rows + 2)]
        self.Omega = self.Omega.expand(rows + 2, columns + 2, indices_of_original_elements,
                                       indices_of_original_elements)  # CHECK
        self.Xi = self.Xi.expand(rows + 2, 1, indices_of_original_elements, [0])

        # update Omega diagonals
        for b in range(4):
            self.Omega.value[b][b] += 1.0 / motion_noise

        # update remaining matrix entries
        for b in range(2):
            self.Omega.value[b][b + 2] += -1.0 / motion_noise
            self.Omega.value[b + 2][b] += -1.0 / motion_noise
            self.Xi.value[b][0] += -distance_update[b] / motion_noise
            self.Xi.value[b + 2][0] += distance_update[b] / motion_noise

        # reshape Omega and Xi for online SLAM
        newlist = range(2, len(self.Omega.value))
        a = self.Omega.take([0, 1], newlist)
        b = self.Omega.take([0, 1])
        c = self.Xi.take([0, 1], [0])
        self.Omega = self.Omega.take(newlist) - a.transpose() * b.inverse() * a
        self.Xi = self.Xi.take(newlist, [0]) - a.transpose() * b.inverse() * c

        # calculate mu and retrieve x, y
        mu = self.Omega.inverse() * self.Xi
        x,y = mu.value[0][0], mu.value[1][0]

        return x, y


class WayPointPlanner:
    """Create a planner to navigate the rover to reach all the intended way points from an unknown start position.
    """
    def __init__(self,  max_distance, max_steering):
        """Initialize your planner here.

        Args:
            max_distance(float): the max distance the robot can travel in a single move.
            max_steering(float): the max steering angle the robot can turn in a single move.
        """
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.slam = SLAM()
        self.x = 0
        self.y = 0
        self.target_ID = None
        self.target = None
        self.spiral_distance_unit = 1
        self.spiral_index = 0
        self.spiral_distance_covered = 0
        self.checked_beacons = []
        self.end_of_segment = False

    def next_move(self, sample_todo, measurements):
        """Next move based on the current set of measurements.

        Args:
            sample_todo(list): Set of locations remaining still needing a sample to be taken.
            measurements(dict): Collection of measurements from beacons and test sites in range.
                in the format {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'beacon'}, ...}

        Return:
            Next command to execute on the rover.
                allowed:
                    'move 1.570963 1.0' - turn left 90 degrees and move 1.0 distance
                    'sample' - take sample (will succeed if within tolerance of intended sample site)
        """

        # if robot detects a site, move toward it.  otherwise, spiral until a site is detected
        site_detected = False
        spiral = True
        nearest_beacon = None
        nearest_beacon_ID = None
        nearest_beacon_distance = 10000

        for id, input_dict in measurements.iteritems():
            distance = input_dict['distance']
            bearing = input_dict['bearing']
            m_type = input_dict['type']

            # if the landmark is a site, make sure it's the closest, then set as target
            if m_type == 'site':
                spiral = False
                # if another site has already been detected.  test for which one is closer.
                if site_detected:
                    if distance < self.target[self.target_ID]['distance']:
                        self.target_ID = id
                        self.target = {id: input_dict}
                # if this is the first site detected, set it as target
                else:
                    site_detected = True
                    self.target_ID = id
                    self.target = {id: input_dict}
            else:
                if id not in self.checked_beacons and distance < nearest_beacon_distance:
                    spiral = False
                    nearest_beacon = {id: input_dict}
                    nearest_beacon_ID = id
                    nearest_beacon_distance = distance

        # if a site has been detected, move toward it
        if site_detected:
            # if within range to sample, sample
            if self.target[self.target_ID]['distance'] < 0.25:
                action = 'sample'
                self.target = None
                self.target_ID = None
            # if too far to sample, move toward it, respecting max_distance and max_steering
            else:
                truncated_distance = min(self.max_distance, self.target[self.target_ID]['distance'])
                truncated_steering = truncate_angle(self.target[self.target_ID]['bearing'])
                truncated_steering = max(-self.max_steering, truncated_steering)
                truncated_steering = min(self.max_steering, truncated_steering)
                action = 'move ' + str(truncated_steering) + ' ' + str(truncated_distance)
                #update robot position
                self.x, self.y = self.slam.process_movement(truncated_steering, truncated_distance)

        # if the beacons all haven't been checked, check them, starting with the closest
        if not site_detected and not spiral:
            if nearest_beacon_distance < 0.25:
                self.checked_beacons.append(nearest_beacon_ID)
                action = 'move ' + str(0) + ' ' + str(0)
            else:
                truncated_distance = min(self.max_distance, nearest_beacon[nearest_beacon_ID]['distance'])
                truncated_steering = truncate_angle(nearest_beacon[nearest_beacon_ID]['bearing'])
                truncated_steering = max(-self.max_steering, truncated_steering)
                truncated_steering = min(self.max_steering, truncated_steering)
                action = 'move ' + str(truncated_steering) + ' ' + str(truncated_distance)
                self.x, self.y = self.slam.process_movement(truncated_steering, truncated_distance)

        if spiral:
            # if spiral straights are longer than max distance, break moves along straight into segments of length max_distance

            if self.spiral_distance_unit > self.max_distance:
                # if the next action will terminate a spiral straight segment, travel the remaining distance then start a new segment
                distance_remaining = self.spiral_distance_unit - self.spiral_distance_covered
                if distance_remaining < self.max_distance:
                    move_distance = distance_remaining
                    move_steering = 0
                    self.spiral_distance_covered = 0
                    self.spiral_index += 1
                    self.end_of_segment = True
                # if not yet at the end of a spiral straight, keep moving straight max_distance units
                else:
                    if self.end_of_segment == True:
                        move_steering = pi / 2
                    else:
                        move_steering = 0
                    move_distance = self.max_distance
                    self.spiral_distance_covered += self.max_distance
                    self.end_of_segment = False

            # if the length of each spiral straight is less than max_distance, spiral normally
            else:
                move_steering = pi / 2
                move_distance = self.spiral_distance_unit
                self.spiral_index += 1

            # if the robot has moved straight twice using the same length of spiral segment, increment the segment length
            if self.spiral_index == 2:
                self.spiral_index = 0
                self.spiral_distance_unit += 1.75

            # define action
            action = 'move ' + str(move_steering) + ' ' + str(move_distance)
            # update robot position and increment spiral_index
            self.x, self.y = self.slam.process_movement(move_steering, move_distance)

        return action