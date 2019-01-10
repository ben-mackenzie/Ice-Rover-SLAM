#!/usr/bin/python

import unittest
import multiprocessing as mproc
import numpy as np
import random
import math
import time
import traceback
import hashlib

import ice_rover
import robot

PI = math.pi

########################################################################
# for debugging set the time limit to a big number
########################################################################
TIME_LIMIT = 5  # seconds

########################################################################
# set to True for lots-o-output, also passed into robot under test
########################################################################
VERBOSE_FLAG = True

########################################################################
# TODO: you can set NOISE_FLAG to false during development
# but be sure to run and test with noise = True
# before submitting your solution.
# For running the example static test cases defined in __init__() you should 
# set to False since there is no online adaptation for them.
########################################################################
NOISE_FLAG = True
NOISE_MOVE = 0.01

########################################################################
# used to generate unique ids for landmarks.  will change for grader
########################################################################
HASH_SEED = 'some_seed'

PART_A_CREDIT = 0.33
PART_B_CREDIT = 0.67


class Submission:
    """Student Submission.

    Attributes:
        submission_waypoints(Queue): Student score of last executed plan.
        submission_error(Queue): Error messages generated during last executed plan.
    """

    def __init__(self):
        self.submission_waypoints = mproc.Queue(1)
        self.submission_error = mproc.Queue(1)

    def _reset(self):
        """Reset submission results.
        """
        while not self.submission_waypoints.empty():
            self.submission_waypoints.get()

        while not self.submission_error.empty():
            self.submission_error.get()

    def execute_student_plan(self, area_map, sample_todo, max_distance=1.0, max_steering=PI/2.+0.01,
                             horizon_distance=3.0):
        """Execute student plan and store results in submission.

        Args:
            area_map(list(list)): the area map to test against.
            sample_todo(list): the order of boxes to deliver.
            max_distance(float): maximum distance per move.
            max_steering(float): maximum steering per move.
            horizon_distance(float): maximum range of sensors.
        """
        self._reset()

        state = State(area_map, sample_todo, max_distance, max_steering, horizon_distance)

        if VERBOSE_FLAG:
            print('Initial State:')
            print(state)

        try:
            student_planner = ice_rover.WayPointPlanner(max_distance, max_steering)

            state_output = ''

            while len(state.reached_locations) < len(sample_todo):
                state_output += str(state)

                action = student_planner.next_move(state.sample_todo, state.generate_measurements())
                state.update_according_to(action)

                if VERBOSE_FLAG:
                    print(state)

            if VERBOSE_FLAG:
                print('Final State:')
                print(state)

            self.submission_waypoints.put(state.reached_locations)

        except:
            self.submission_error.put(traceback.format_exc())
            self.submission_waypoints.put([])


class State:
    """Current State.

    Args:
        area_map(list(list)): the area map.
        sample_todo(set(list)): the list of areas to visit and sample
        max_distance(float): the max distance the robot can travel in a single move.
        max_steering(float): the max steering angle the robot can turn in a single move.
        horizon_distance(float): maximum range of sensors.

    Attributes:
        sample_todo(set(list)): the list of locations to visit and sample
        reached_locations(list): way points successfully reached.
        max_distance(float): max distance the robot can travel in one move.
        max_steering(float): the max steering angle the robot can turn in a single move.
        landmarks(list): landmarks to ping for location
        _start_position(dict): location of initial robot placement
    """
    ROBOT_RADIUS = 0.25
    SAMPLE_DISTANCE = 0.25
    WAIT_PENALTY = 0.1  # seconds

    def __init__(self, area_map, sample_todo=list(), max_distance=1.0, max_steering=PI/2.+0.01, horizon_distance=3.0):
        self.orig_todo = list(sample_todo)
        self.sample_todo = list(sample_todo)
        self.reached_locations = list()
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.horizon_distance = horizon_distance

        rows = len(area_map)
        cols = len(area_map[0])

        self.landmarks = list()
        self._start_position = dict()

        # Now process the interior of the provided map
        for i in range(rows):
            for j in range(cols):
                this_square = area_map[i][j]
                x, y = float(j), -float(i)

                # Process landmarks
                if this_square == 'L':
                    landmark = dict()
                    landmark['x'] = x
                    landmark['y'] = y

                    self.landmarks.append(landmark)

                # Process start
                if this_square == '@':
                    self._start_position['x'] = x + 0.5
                    self._start_position['y'] = y - 0.5

        # initialize the robot at the start position and at a bearing pointing due east
        self.robot = robot.Robot(x=self._start_position['x'], y=self._start_position['y'], bearing=0.0,
                                 max_distance=self.max_distance, max_steering=self.max_steering)

    def generate_measurements(self, noise=NOISE_FLAG):
        """Generate measurements of landmarks within range.

        Args:
            noise(bool): Move with noise if True.
                Default: NOISE_FLAG

        Returns:
            Measurements to landmarks in the format:
                {'landmark id':{'distance':0.0, 'bearing':0.0}, ...}
        """
        measurements = dict()

        # process landmarks
        for location in self.landmarks:
            distance, bearing = self.robot.measure_distance_and_bearing_to((location['x'], location['y']), noise=noise)

            if distance <= self.horizon_distance:
                measurements[int(hashlib.md5(str(location) + HASH_SEED).hexdigest(), 16)] = {'distance': distance,
                                                                                             'bearing': bearing,
                                                                                             'type': 'beacon'}
        # process sample sites
        for location in self.sample_todo:
            distance, bearing = self.robot.measure_distance_and_bearing_to(location, noise=noise)

            if distance <= self.horizon_distance:
                measurements[int(hashlib.md5(str(location) + HASH_SEED).hexdigest(), 16)] = {'distance': distance,
                                                                                             'bearing': bearing,
                                                                                             'type': 'site'}

        return measurements

    def update_according_to(self, action, noise=NOISE_FLAG):
        """Update state according to action.

        Args:
            action(str): action to execute.
            noise(bool): Move with noise if True.
                Default: NOISE_FLAG

        Raises:
            Exception: if improperly formatted action.
        """
        action = action.split()
        action_type = action[0]

        if action_type == 'move':
            steering, distance = action[1:]
            self._attempt_move(float(steering), float(distance), noise=noise)

        elif action_type == 'sample':
            self._attempt_sample()

        else:
            # improper move format: kill test
            raise Exception('improperly formatted action: {}'.format(''.join(action)))

    def _attempt_move(self, steering, distance, noise=NOISE_FLAG):
        """Attempt move action if valid.

        The robot may move between 0 and max_distance
        The robot may turn between -max_steering and +max_steering

        Illegal moves - the robot will not move
        - Moving a distance outside of [0,max_distance]
        - Steering angle outside [-max_steering, max_steering]

        Args:
            steering(float): Angle to turn before moving.
            distance(float): Distance to travel.

        Raises:
            ValueError: if improperly formatted move destination.
        """
        try:
            distance_ok = 0.0 <= distance <= self.max_distance
            steering_ok = (-self.max_steering) <= steering <= self.max_steering

            if noise:
                steering += random.uniform(-NOISE_MOVE, NOISE_MOVE)
                distance *= random.uniform(1.0 - NOISE_MOVE, 1.0 + NOISE_MOVE)

            if distance_ok and steering_ok:
                self.robot.move(steering, distance)

        except ValueError:
            raise Exception('improperly formatted move command : {} {}'.format(steering, distance))

    def _attempt_sample(self):
        """Attempt to take sample if near any sample point not already taken.

        Take sample if current location is within SAMPLE_DISTANCE of a required sample site.
        Otherwise, pause for WAIT_PENALTY
        """
        for sample_location in self.sample_todo:
            distance = np.sqrt((self.robot.x - sample_location[0]) ** 2 + (self.robot.y - sample_location[1]) ** 2)
            if distance < self.SAMPLE_DISTANCE:
                self.reached_locations.append(sample_location)
                self.sample_todo.remove(sample_location)
                return

        time.sleep(self.WAIT_PENALTY)

        if VERBOSE_FLAG:
            print "*** Location ({}, {}) is not a requested sample location.".format(self.robot.x, self.robot.y)

    def __repr__(self):
        """Output state object as string.
        """
        output = '\n'
        output += 'Robot State:\n'
        output += '\t x = {:6.2f}, y = {:6.2f}, hdg = {:6.2f}\n'.format(self.robot.x, self.robot.y,
                                                                        self.robot.bearing * 180. / PI)
        output += 'Locations Reached: {}\n'.format(self.reached_locations)
        output += 'Remaining Locations: {}\n'.format(self.sample_todo)

        return output


class PartATestCase(unittest.TestCase):
    """Test PartA
    """
    results_file = 'results_partA.txt'

    results = ['', 'PART A TEST CASE RESULTS']
    SCORE_TEMPLATE = "\n".join((
        "\n-----------",
        "Test Case {test_case}",
        "  Expected Location: {expected}",
        "  SLAM Location: {location}",
        "  Credit: {score:.0%}"
    ))
    FAIL_TEMPLATE = "\n".join((
        "\n-----------",
        "Test Case {test_case}",
        "  Failed: {message}",
        "  Credit: 0.0"
    ))

    credit = []

    @classmethod
    def tearDownClass(cls):
        """Save student results at conclusion of test.
        """
        # Prints results after all tests complete
        with open(cls.results_file, 'w') as f:
            for line in cls.results:
                f.write(line)
            f.write("\n-----------")
            f.write('\nPart A Credit: {:.2f}'.format(sum(cls.credit)/len(cls.credit)*100.0 * PART_A_CREDIT))

        try:
            grade = sum(cls.credit) / len(cls.credit) * 100.0 * PART_A_CREDIT

        except ZeroDivisionError:
            grade = 0.0

        print('\nPart A Credit: {:.2f}'.format(grade))

    def run_with_params(self, params):
        """Run test case using desired parameters.
        Args:
            params(dict): a dictionary of test parameters.
        """

        state = State(params['area_map'])
        rover_slam = ice_rover.SLAM()

        for move in params['move']:     

            meas = state.generate_measurements()
            belief = rover_slam.process_measurements(meas)
            truth = (state.robot.x - state._start_position['x'], state.robot.y - state._start_position['y'])                      

            action = move.split()
            state.update_according_to(move)
            rover_slam.process_movement(float(action[1]), float(action[2]))

            dist_error = robot.compute_distance(belief, truth)
            if VERBOSE_FLAG:
                print "Current Belief:", belief
                print "True Position:", truth
                print "Error:", dist_error, "\n"

        if dist_error < params['test_tolerance']:
            self.results.append(self.SCORE_TEMPLATE.format(expected=truth,
                                                           location=belief, score=1.0, **params))
            score = 1.0
            self.credit.append(score)
        else:
            self.results.append(self.FAIL_TEMPLATE.format(message='Distance greater than tolerance {}'
                                                          .format(params['test_tolerance']), **params))
            score = 0.0
            self.credit.append(score)
            self.assertTrue(dist_error < params['test_tolerance'],
                            'Location error {} as a distance must be less than {}'.format(dist_error,
                                                                                          params['test_tolerance']))

        print('test case {}: {:.0%}'.format(params['test_case'], score))

    def test_case1(self):
        params = {'test_case': 1,
                  'area_map': ['......',
                               '..LL..',
                               '..L@..'],
                  'move': ['move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0'],
                  'test_tolerance': 0.1}

        self.run_with_params(params)

    def test_case2(self):
        params = {'test_case': 2,
                  'area_map': ['LLLLLLLL',
                               'L......L',
                               'L...@..L',
                               'L......L',
                               'LLLLLLLL'],
                  'move': ['move 0.0 1.0',
                           'move 0.0 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0'],
                  'test_tolerance': 0.1}

        self.run_with_params(params)

    def test_case3(self):
        params = {'test_case': 3,
                  'area_map': ['LLLLLLLL',
                               'L@.....L',
                               'L..LL..L',
                               'L......L',
                               'LLLLLLLL'],
                  'move': ['move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0'],
                  'test_tolerance': 0.1}

        self.run_with_params(params)

    def test_case4(self):
        params = {'test_case': 4,
                  'area_map': ['LLLLLLLL',
                               'L..L..@L',
                               'L......L',
                               'L..LLLLL',
                               'L..L...L',
                               'L......L',
                               'LLLLLLLL'],
                  'move': ['move 1.570963 0.0', 'move 1.570963 0.0',
                           'move 0.785481 1.0', 'move 0.0 0.5',
                           'move 1.570963 0.0', 'move 1.570963 0.0',
                           'move 0.785481 1.0', 'move 0.0 0.5'],
                  'test_tolerance': 0.1}

        self.run_with_params(params)


class PartBTestCase(unittest.TestCase):
    """ Test PartB.
    """
    results_file = 'results_partB.txt'

    results = ['', 'PART B TEST CASE RESULTS']
    SCORE_TEMPLATE = "\n".join((
        "\n-----------",
        "Test Case {test_case}",
        "  Expected locations: {todo}",
        "  Reached locations: {placed}",
        "  Credit: {score:.0%}"
    ))
    FAIL_TEMPLATE = "\n".join((
        "\n-----------",
        "Test Case {test_case}",
        "  Failed: {message}",
        "  Credit: 0.0"
    ))

    credit = []

    def setUp(self):
        """Initialize test setup.
        """
        self.student_submission = Submission()

    @classmethod
    def tearDownClass(cls):
        """Save student results at conclusion of test.
        """
        # Prints results after all tests complete
        with open(cls.results_file, 'w') as f:
            for line in cls.results:
                f.write(line)
            f.write("\n-----------")
            if len(cls.credit) != 0:
                f.write('\nPart B Credit: {:.2f}'.format(sum(cls.credit) / len(cls.credit)))

        try:
            grade = sum(cls.credit) / len(cls.credit) * 100.0 * PART_B_CREDIT

        except ZeroDivisionError:
            grade = 0.0

        print('\nPart B Credit: {:.2f}'.format(grade))

    def run_with_params(self, params):
        """Run test case using desired parameters.
        Args:
            params(dict): a dictionary of test parameters.
        """
        test_process = mproc.Process(target=self.student_submission.execute_student_plan,
                                     args=(params['area_map'],
                                           params['todo'],
                                           params['max_distance'],
                                           params['max_steering'],
                                           params['horizon_distance']))

        error_message = ''
        sampled_sites = 0
        score = 0.0

        try:
            test_process.start()
            test_process.join(TIME_LIMIT)
        except Exception as exp:
            error_message = exp.message

        # If test still running then terminate
        if test_process.is_alive():
            test_process.terminate()
            error_message = ('Test aborted due to timeout. ' +
                             'Test was expected to finish in fewer than {} second(s).'.format(TIME_LIMIT))
            self.results.append(self.FAIL_TEMPLATE.format(message=error_message, **params))
            score = 0.0
            self.credit.append(0.0)
        else:
            # Get number of sites visited
            if not self.student_submission.submission_waypoints.empty():
                sampled_sites = self.student_submission.submission_waypoints.get()

            score = len(sampled_sites) / float(len(params['todo']))

            if not self.student_submission.submission_error.empty():
                error_message = self.student_submission.submission_error.get()
                self.results.append(self.FAIL_TEMPLATE.format(message=error_message, **params))

            else:
                self.results.append(self.SCORE_TEMPLATE.format(placed=sampled_sites, score=score, **params))

            self.credit.append(score)

        print('test case {} credit: {:.0%}'.format(params['test_case'], score))
        if error_message:
            print('{}'.format(error_message))

        self.assertFalse(error_message, error_message)
        self.assertTrue(round(score, 7) == 1.0,
                        "Only {} way points were reached out of the {} requested.".format(sampled_sites,
                                                                                          len(params['todo'])))

    def test_case1(self):
        params = {'test_case': 1,
                  'area_map': ['LLLLLLLL',
                               'L..L...L',
                               'L......L',
                               'L..L...L',
                               'L......L',
                               'L.....@L',
                               'LLLLLLLL'],
                  'todo': [(1.5, -1.5),
                           (5.0, -3.5)],
                  'horizon_distance': 4.0,
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01}

        self.run_with_params(params)

    def test_case2(self):
        params = {'test_case': 2,
                  'area_map': ['LLLLLLLL',
                               'L..L..@L',
                               'L......L',
                               'L..LLLLL',
                               'L..L...L',
                               'L......L',
                               'LLLLLLLL'],
                  'todo': [(4.5, -4.5),
                           (1.5, -2.0)],
                  'horizon_distance': 4.0,
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01}

        self.run_with_params(params)

    def test_case3(self):
        params = {'test_case': 3,
                  'area_map': ['LLLLLLLL',
                               'L..L...L',
                               'L......L',
                               'L..LLLLL',
                               'L..L..LL',
                               'L.....@L',
                               'LLLLLLLL'],
                  'todo': [(5.5, -2.0),
                           (4.5, -4.5)],
                  'horizon_distance': 4.0,
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01}

        self.run_with_params(params)

    def test_case4(self):
        params = {'test_case': 4,
                  'area_map': ['LLLLLLLLL',
                               'L@.L...LL',
                               'LL...L..L',
                               'LLLLLLLLL'],
                  'todo': [(5.5, -1.5),
                           (7.0, -2.5)],
                  'horizon_distance': 4.0,
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01}

        self.run_with_params(params)

    def test_case5(self):
        params = {'test_case': 5,
                  'area_map': ['LLLLLLL',
                               'LL.@.LL',
                               'L..L..L',
                               'LLLLLLL'],
                  'todo': [(1.5, -2.5),
                           (5.5, -2.5)],
                  'horizon_distance': 4.0,
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01}

        self.run_with_params(params)

    def test_case6(self):
        params = {'test_case': 6,
                  'area_map': ['LLLLLLL',
                               'LL.@.LL',
                               'LLLLLLL'],
                  'todo': [(2.5, -1.5),
                           (4.5, -1.5)],
                  'horizon_distance': 4.0,
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01}

        self.run_with_params(params)

    def test_case7(self):
        params = {'test_case': 7,
        'area_map': ['.....................................................',
                     '..LLLLLL..LLLLLLL..L.....L..LLLLLLLL..................',
                     '..L....L.....L.....L.....L..L.......L.................',
                     '..L....L.....L.....L.....L..L.......L.................',
                     '..L....L.....L.....LLLLLLL..L..LLLLL..................',
                     '..LLLLLL.....L...........L..L...L.....................',
                     '..L....L.....L...........L..L.....L...................',
                     '..L....L.....L...........L..L......L..................',
                     'LLL....L..LLLLLLL........L..L.......LLLLLLLLLLLLLLLL..',
                     '@L....................................................'],
        'todo': [(4.0, -3.0), (53.0, -10.0), (32.0, -4.0), (7.0, -7.0), (53, 53)],
        'horizon_distance': 3.0,
        'max_distance': 1.0,
        'max_steering': PI / 2. + 0.01}
        self.run_with_params(params)

# Only run all of the test automatically if this file was executed from the command line.
# Otherwise, let Nose/py.test do it's own thing with the test cases.
if __name__ == "__main__":
    all_suites = map(lambda x: unittest.TestLoader().loadTestsFromTestCase(x), [PartATestCase, PartBTestCase])
    all_tests = unittest.TestSuite(all_suites)
    unittest.TextTestRunner(verbosity=2).run(all_tests)
