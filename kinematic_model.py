from math import cos, sin, tan, atan2


class KinematicBicycleModel:
    def __init__(
        self,
        wheelbase: float,
        max_steer: float,
        dt: float = 0.05,
        c_r: float = 0.0,
        c_a: float = 0.0,
    ):

        """
        Constructor for the 2D Kinematic Bicycle Model

        :param wheelbase:           (float) vehicle's wheelbase [m]
        :param max_steer:           (float) vehicle's steering limits [rad]
        :param dt:                  (float) discrete time period [s]
        :param c_r:                 (float) vehicle's coefficient of resistance
        :param c_a:                 (float) vehicle's aerodynamic coefficient
        """

        self.dt = dt
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.c_r = c_r
        self.c_a = c_a

    def kinematic_model(
        self,
        x: float,
        y: float,
        yaw: float,
        velocity: float,
        throttle: float,
        steering_angle: float,
    ) -> tuple[float, ...]:

        """
        This method should updated at every time step

        :param x:                   (float) vehicle's x-coordinate [m]
        :param y:                   (float) vehicle's y-coordinate [m]
        :param yaw:                 (float) vehicle's heading [rad]
        :param velocity:            (float) vehicle's velocity in the x-axis [m/s]
        :param throttle:            (float) vehicle's accleration [m/s^2]
        :param delta:               (float) vehicle's steering angle [rad]

        :return new_x:              (float) vehicle's x-coordinate [m]
        :return new_y:              (float) vehicle's y-coordinate [m]
        :return new_yaw:            (float) vehicle's heading [rad]
        :return new_velocity:       (float) vehicle's velocity in the x-axis [m/s]
        :return steering_angle:     (float) vehicle's steering angle [rad]
        :return angular_velocity:   (float) vehicle's angular velocity [rad/s]
        """

        # Compute the local velocity in the x-axis
        friction = velocity * (self.c_r + self.c_a * velocity)
        new_velocity = velocity + self.dt * (throttle - friction)

        # Limit steering angle to physical vehicle limits
        steering_angle = (
            -self.max_steer
            if steering_angle < -self.max_steer
            else self.max_steer
            if steering_angle > self.max_steer
            else steering_angle
        )

        # Compute the angular velocity
        angular_velocity = velocity * tan(steering_angle) / self.wheelbase

        # Compute the final state using the discrete time model
        new_x = x + velocity * cos(yaw) * self.dt
        new_y = y + velocity * sin(yaw) * self.dt
        new_yaw = atan2(
            sin(yaw + angular_velocity * self.dt), cos(yaw + angular_velocity * self.dt)
        )  # normalise angle
        return new_x, new_y, new_yaw, new_velocity, steering_angle, angular_velocity
