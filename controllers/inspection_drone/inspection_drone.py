"""inspection_drone controller."""

from controller import Robot
import math

import sys 
try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")

CLOCKWISE = 1
COUNTER_CLOCKWISE = -1

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


class Mavic (Robot):
    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 30.0          # P constant of the pitch PID.

    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1
    # Precision between the target position and the robot position in meters
    target_precision = 0.5

    def __init__(self):
        Robot.__init__(self)

        self.time_step = int(self.getBasicTimeStep())

        # Get and enable devices.
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)
        self.lidar = self.getDevice("lidar_360")
        self.lidar.enable(self.time_step)

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        self.current_pose = 6 * [0]  # X, Y, Z, yaw, pitch, roll
        self.target_position = [0, 0, 0]
        self.target_index = 0
        self.target_altitude = 0

        self.avoidance_state = "NORMAL"
        self.avoidance_direction = CLOCKWISE
        self.avoidance_yaw = 0.0
        self.bypass_start_position = None

    def set_position(self, pos):
        """
        Set the new absolute position of the robot
        Parameters:
            pos (list): [X,Y,Z,yaw,pitch,roll] current absolute position and angles
        """
        self.current_pose = pos

    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=False):
        """
        Move the robot to the given coordinates
        Parameters:
            waypoints (list): list of X,Y coordinates
            verbose_movement (bool): whether to print remaning angle and distance or not
            verbose_target (bool): whether to print targets or not
        Returns:
            yaw_disturbance (float): yaw disturbance (negative value to go on the right)
            pitch_disturbance (float): pitch disturbance (negative value to go forward)
        """

        if self.target_position[0:2] == [0, 0]:  # Initialization
            self.target_position[0:2] = waypoints[0]
            if verbose_target:
                print("First target: ", self.target_position[0:2])

        # if the robot is at the position with a precision of target_precision
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):

            self.target_index += 1
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
            self.target_position[0:2] = waypoints[self.target_index]
            if verbose_target:
                print("Target reached! New target: ",
                      self.target_position[0:2])

        # This will be in ]-pi;pi]
        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        # This is now in ]-2pi;2pi[
        angle_left = self.target_position[2] - self.current_pose[5]
        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi

        # Turn the robot to the left or to the right according the value and the sign of angle_left
        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        # non proportional and decreasing function
        pitch_disturbance = clamp(
            np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        if verbose_movement:
            distance_left = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + (
                (self.target_position[1] - self.current_pose[1]) ** 2))
            print("remaning angle: {:.4f}, remaning distance: {:.4f}".format(
                angle_left, distance_left))
        return yaw_disturbance, pitch_disturbance

    def minimum_valid_range(self, values, minimum, maximum):
        valid = [
            value for value in values
            if math.isfinite(value)
            and minimum <= value <= maximum
        ]

        return min(valid) if valid else float("inf")


    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def run(self):
        t1 = self.getTime()

        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0

        # Start with short, nearby waypoints.
        waypoints = [
            [-45, 5],
            [-55, 5],
            [-55, 15],
            [-45, 15]
        ]

        self.target_altitude = 2.0

        while self.step(self.time_step) != -1:

            # Read flight sensors.
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x_pos, y_pos, altitude = self.gps.getValues()

            roll_acceleration, pitch_acceleration, _ = (
                self.gyro.getValues()
            )

            self.set_position([
                x_pos,
                y_pos,
                altitude,
                roll,
                pitch,
                yaw
            ])

            # Read LiDAR.
            ranges = self.lidar.getRangeImage()
            scan_size = len(ranges)

            minimum_range = self.lidar.getMinRange()
            maximum_range = self.lidar.getMaxRange()

            # For a cylindrical 360-degree Webots LiDAR, the middle
            # of the range image should face forward.
            front_center = scan_size // 2
            front_half_width = scan_size // 12  # 30 degrees each side

            front_values = ranges[
                front_center - front_half_width:
                front_center + front_half_width
            ]

            # Approximate 90-degree sectors on either side.
            left_values = ranges[
                scan_size // 8:
                3 * scan_size // 8
            ]

            right_values = ranges[
                5 * scan_size // 8:
                7 * scan_size // 8
            ]

            front_distance = self.minimum_valid_range(
                front_values,
                minimum_range,
                maximum_range
            )

            left_distance = self.minimum_valid_range(
                left_values,
                minimum_range,
                maximum_range
            )

            right_distance = self.minimum_valid_range(
                right_values,
                minimum_range,
                maximum_range
            )

            WARNING_DISTANCE = 3.5
            BYPASS_DISTANCE = 3.0

            if altitude > self.target_altitude - 1.0:

                # -----------------------------------------
                # Normal waypoint navigation
                # -----------------------------------------

                if self.avoidance_state == "NORMAL":
                    if front_distance < WARNING_DISTANCE:
                        pitch_disturbance = 0.0

                        # Turn toward the side with more open space.
                        if left_distance > right_distance:
                            self.avoidance_direction = 1
                            print("Obstacle ahead: turning left")
                        else:
                            self.avoidance_direction = -1
                            print("Obstacle ahead: turning right")

                        # Turn approximately 90 degrees.
                        self.avoidance_yaw = self.normalize_angle(
                            yaw
                            + self.avoidance_direction
                            * math.pi / 2.0
                        )

                        self.avoidance_state = "TURN"

                    elif self.getTime() - t1 > 0.1:
                        yaw_disturbance, pitch_disturbance = (
                            self.move_to_target(waypoints)
                        )

                        t1 = self.getTime()

                # -----------------------------------------
                # Turn away from obstacle
                # -----------------------------------------

                elif self.avoidance_state == "TURN":
                    pitch_disturbance = 0.0

                    yaw_error = self.normalize_angle(
                        self.avoidance_yaw - yaw
                    )

                    yaw_disturbance = clamp(
                        0.8 * yaw_error,
                        -self.MAX_YAW_DISTURBANCE,
                        self.MAX_YAW_DISTURBANCE
                    )

                    if abs(yaw_error) < 0.12:
                        self.bypass_start_position = [
                            x_pos,
                            y_pos
                        ]

                        self.avoidance_state = "BYPASS"
                        print("Turn complete: beginning bypass")

                # -----------------------------------------
                # Fly alongside the obstacle
                # -----------------------------------------

                elif self.avoidance_state == "BYPASS":
                    yaw_error = self.normalize_angle(
                        self.avoidance_yaw - yaw
                    )

                    yaw_disturbance = clamp(
                        0.8 * yaw_error,
                        -self.MAX_YAW_DISTURBANCE,
                        self.MAX_YAW_DISTURBANCE
                    )

                    # Negative pitch disturbance moves forward.
                    pitch_disturbance = -0.5

                    distance_traveled = math.hypot(
                        x_pos - self.bypass_start_position[0],
                        y_pos - self.bypass_start_position[1]
                    )

                    # Stop if another obstacle appears.
                    if front_distance < 1.5:
                        pitch_disturbance = 0.0

                        self.avoidance_yaw = self.normalize_angle(
                            yaw
                            + self.avoidance_direction
                            * math.pi / 2.0
                        )

                        self.avoidance_state = "TURN"

                    elif distance_traveled >= BYPASS_DISTANCE:
                        pitch_disturbance = 0.0
                        yaw_disturbance = 0.0
                        self.avoidance_state = "NORMAL"

                        print("Bypass complete: resuming waypoint")

            # Flight stabilization.
            roll_input = (
                self.K_ROLL_P * clamp(roll, -1, 1)
                + roll_acceleration
                + roll_disturbance
            )

            pitch_input = (
                self.K_PITCH_P * clamp(pitch, -1, 1)
                + pitch_acceleration
                + pitch_disturbance
            )

            yaw_input = yaw_disturbance

            altitude_error = (
                self.target_altitude
                - altitude
                + self.K_VERTICAL_OFFSET
            )

            clamped_altitude_error = clamp(
                altitude_error,
                -1,
                1
            )

            vertical_input = (
                self.K_VERTICAL_P
                * pow(clamped_altitude_error, 3.0)
            )

            front_left_motor_input = (
                self.K_VERTICAL_THRUST
                + vertical_input
                - yaw_input
                + pitch_input
                - roll_input
            )

            front_right_motor_input = (
                self.K_VERTICAL_THRUST
                + vertical_input
                + yaw_input
                + pitch_input
                + roll_input
            )

            rear_left_motor_input = (
                self.K_VERTICAL_THRUST
                + vertical_input
                + yaw_input
                - pitch_input
                - roll_input
            )

            rear_right_motor_input = (
                self.K_VERTICAL_THRUST
                + vertical_input
                - yaw_input
                - pitch_input
                + roll_input
            )

            self.front_left_motor.setVelocity(
                front_left_motor_input
            )

            self.front_right_motor.setVelocity(
                -front_right_motor_input
            )

            self.rear_left_motor.setVelocity(
                -rear_left_motor_input
            )

            self.rear_right_motor.setVelocity(
                rear_right_motor_input
            )

# To use this controller, the basicTimeStep should be set to 8 and the defaultDamping
# with a linear and angular damping both of 0.5
robot = Mavic()
robot.run()
