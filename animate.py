import csv
import numpy as np
from math import sin, cos, atan2, radians
from random import uniform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from shapely.geometry import LineString, Polygon, LinearRing
import pickle

with open("data/strtree.pickle", "rb") as f:
    rtree = pickle.load(f)

from car_description import CarDescription
from cubic_spline_interpolator import generate_cubic_spline
from kinematic_model import KinematicBicycleModel


class StanleyController:
    def __init__(
        self,
        control_gain=2.5,
        softening_gain=1.0,
        yaw_rate_gain=0.0,
        steering_damp_gain=0.0,
        max_steer=np.deg2rad(24),
        wheelbase=0.0,
        path_x=None,
        path_y=None,
        path_yaw=None,
    ):

        """
        Stanley Controller

        At initialisation
        :param control_gain:                (float) time constant [1/s]
        :param softening_gain:              (float) softening gain [m/s]
        :param yaw_rate_gain:               (float) yaw rate gain [rad]
        :param steering_damp_gain:          (float) steering damp gain
        :param max_steer:                   (float) vehicle's steering limits [rad]
        :param wheelbase:                   (float) vehicle's wheelbase [m]
        :param path_x:                      (numpy.ndarray) list of x-coordinates along the path
        :param path_y:                      (numpy.ndarray) list of y-coordinates along the path
        :param path_yaw:                    (numpy.ndarray) list of discrete yaw values along the path
        :param dt:                          (float) discrete time period [s]

        At every time step
        :param x:                           (float) vehicle's x-coordinate [m]
        :param y:                           (float) vehicle's y-coordinate [m]
        :param yaw:                         (float) vehicle's heading [rad]
        :param target_velocity:             (float) vehicle's velocity [m/s]
        :param steering_angle:              (float) vehicle's steering angle [rad]

        :return limited_steering_angle:     (float) steering angle after imposing steering limits [rad]
        :return target_index:               (int) closest path index
        :return crosstrack_error:           (float) distance from closest path index [m]
        """

        self.k = control_gain
        self.k_soft = softening_gain
        self.k_yaw_rate = yaw_rate_gain
        self.k_damp_steer = steering_damp_gain
        self.max_steer = max_steer
        self.wheelbase = wheelbase

        self.px = path_x
        self.py = path_y
        self.pyaw = path_yaw

    def find_target_path_id(self, x, y, yaw):

        # Calculate position of the front axle
        fx = x + self.wheelbase * cos(yaw)
        fy = y + self.wheelbase * sin(yaw)

        dx = fx - self.px  # Find the x-axis of the front axle relative to the path
        dy = fy - self.py  # Find the y-axis of the front axle relative to the path

        d = np.hypot(dx, dy)  # Find the distance from the front axle to the path
        target_index = np.argmin(d)  # Find the shortest distance in the array

        return target_index, dx[target_index], dy[target_index], d[target_index]

    def calculate_yaw_term(self, target_index, yaw):

        yaw_error = atan2(
            sin(self.pyaw[target_index] - yaw), cos(self.pyaw[target_index] - yaw)
        )

        return yaw_error

    def calculate_crosstrack_term(self, target_velocity, yaw, dx, dy, absolute_error):

        front_axle_vector = np.array([sin(yaw), -cos(yaw)])
        nearest_path_vector = np.array([dx, dy])
        crosstrack_error = (
            np.sign(nearest_path_vector @ front_axle_vector) * absolute_error
        )

        crosstrack_steering_error = atan2(
            (self.k * crosstrack_error), (self.k_soft + target_velocity)
        )

        return crosstrack_steering_error, crosstrack_error

    def calculate_yaw_rate_term(self, target_velocity, steering_angle):

        yaw_rate_error = (
            self.k_yaw_rate * (-target_velocity * sin(steering_angle)) / self.wheelbase
        )

        return yaw_rate_error

    def calculate_steering_delay_term(
        self, computed_steering_angle, previous_steering_angle
    ):

        steering_delay_error = self.k_damp_steer * (
            computed_steering_angle - previous_steering_angle
        )

        return steering_delay_error

    def stanley_control(self, x, y, yaw, target_velocity, steering_angle):

        target_index, dx, dy, absolute_error = self.find_target_path_id(x, y, yaw)
        yaw_error = self.calculate_yaw_term(target_index, yaw)
        crosstrack_steering_error, crosstrack_error = self.calculate_crosstrack_term(
            target_velocity, yaw, dx, dy, absolute_error
        )
        yaw_rate_damping = self.calculate_yaw_rate_term(target_velocity, steering_angle)

        desired_steering_angle = (
            yaw_error + crosstrack_steering_error + yaw_rate_damping
        )

        # Constrains steering angle to the vehicle limits
        desired_steering_angle += self.calculate_steering_delay_term(
            desired_steering_angle, steering_angle
        )
        limited_steering_angle = np.clip(
            desired_steering_angle, -self.max_steer, self.max_steer
        )

        return limited_steering_angle, target_index, crosstrack_error


class Simulation:
    def __init__(self):

        fps = 50.0

        self.dt = 1 / fps
        self.map_size_x = 70
        self.map_size_y = 40
        self.frames = 1050  # 2500
        self.loop = False


class Path:
    def __init__(self):

        # Get path to waypoints.csv
        with open("data/waypoints.csv", newline="") as f:
            rows = list(csv.reader(f, delimiter=","))

        ds = 0.05
        x, y = [[float(i) for i in row] for row in zip(*rows[1:])]
        self.px, self.py, self.pyaw, _ = generate_cubic_spline(x, y, ds)


class Car:
    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, dt):

        # Model parameters
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.v = 0.0
        self.delta = 0.0
        self.omega = 0.0
        self.wheelbase = 2.96
        self.max_steer = radians(33)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 2.0

        # Tracker parameters
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.k = 8.0
        self.ksoft = 1.0
        self.kyaw = 0.01
        self.ksteer = 0.0
        self.crosstrack_error = None
        self.target_id = None

        # Description parameters
        self.overall_length = 4.97
        self.overall_width = 1.964
        self.tyre_diameter = 0.4826
        self.tyre_width = 0.265
        self.axle_track = 1.7
        self.rear_overhang = 0.5 * (self.overall_length - self.wheelbase)
        self.colour = "black"

        self.tracker = StanleyController(
            self.k,
            self.ksoft,
            self.kyaw,
            self.ksteer,
            self.max_steer,
            self.wheelbase,
            self.px,
            self.py,
            self.pyaw,
        )
        self.kbm = KinematicBicycleModel(
            self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a
        )

    def drive(self):

        throttle = uniform(150, 200)
        (
            self.delta,
            self.target_id,
            self.crosstrack_error,
        ) = self.tracker.stanley_control(self.x, self.y, self.yaw, self.v, self.delta)
        self.x, self.y, self.yaw, self.v, _, _ = self.kbm.kinematic_model(
            self.x, self.y, self.yaw, self.v, throttle, self.delta
        )

        print(f"Cross-track term: {self.crosstrack_error}{' '*10}", end="\r")


class Fargs:
    def __init__(
        self,
        ax,
        sim,
        path,
        car,
        car_description,
        car_outline,
        front_right_wheel,
        front_left_wheel,
        rear_right_wheel,
        rear_left_wheel,
        rear_axle,
        annotation,
        target,
    ):

        self.ax = ax
        self.sim = sim
        self.path = path
        self.car = car
        self.car_description = car_description
        self.car_outline = car_outline
        self.front_right_wheel = front_right_wheel
        self.front_left_wheel = front_left_wheel
        self.rear_right_wheel = rear_right_wheel
        self.rear_left_wheel = rear_left_wheel
        self.rear_axle = rear_axle
        self.annotation = annotation
        self.target = target


def animate(frame, fargs):

    ax = fargs.ax
    sim = fargs.sim
    path = fargs.path
    car = fargs.car
    car_description = fargs.car_description
    car_outline = fargs.car_outline
    front_right_wheel = fargs.front_right_wheel
    front_left_wheel = fargs.front_left_wheel
    rear_right_wheel = fargs.rear_right_wheel
    rear_left_wheel = fargs.rear_left_wheel
    rear_axle = fargs.rear_axle
    annotation = fargs.annotation
    target = fargs.target

    # Camera tracks car
    ax.set_xlim(car.x - sim.map_size_x, car.x + sim.map_size_x)
    ax.set_ylim(car.y - sim.map_size_y, car.y + sim.map_size_y)

    # Drive and draw car
    car.drive()
    outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = car_description.plot_car(
        car.x, car.y, car.yaw, car.delta
    )
    car_outline.set_data(*outline_plot)
    front_right_wheel.set_data(*fr_plot)
    rear_right_wheel.set_data(*rr_plot)
    front_left_wheel.set_data(*fl_plot)
    rear_left_wheel.set_data(*rl_plot)
    rear_axle.set_data(car.x, car.y)

    # Show car's target
    target.set_data(path.px[car.target_id], path.py[car.target_id])

    # Annotate car's coordinate above car
    annotation.set_text(f"{car.x:.1f}, {car.y:.1f}")
    annotation.set_position((car.x, car.y + 5))

    plt.title(f"{sim.dt*frame:.2f}s", loc="right")
    plt.xlabel(f"Speed: {car.v:.2f} m/s", loc="left")
    # plt.savefig(f'image/visualisation_{frame:03}.png', dpi=300)

    return (
        car_outline,
        front_right_wheel,
        rear_right_wheel,
        front_left_wheel,
        rear_left_wheel,
        rear_axle,
        target,
    )


def main():

    sim = Simulation()
    path = Path()
    car = Car(path.px[0], path.py[0], path.pyaw[0], path.px, path.py, path.pyaw, sim.dt)
    car_description = CarDescription(
        car.overall_length,
        car.overall_width,
        car.rear_overhang,
        car.tyre_diameter,
        car.tyre_width,
        car.axle_track,
        car.wheelbase,
    )

    interval = sim.dt * 10**3

    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect("equal")

    # add shapely obstacles
    for obs in rtree.geometries:
        if isinstance(obs, Polygon):
            x, y = obs.exterior.xy
        elif isinstance(obs, LinearRing) or isinstance(obs, LineString):
            x, y = obs.xy
        plt.plot(x, y)

    # road = plt.Circle((0, 0), 50, color='gray', fill=False, linewidth=30)
    # ax.add_patch(road)
    ax.plot(path.px, path.py, "--", color="gold")

    empty = ([], [])
    (target,) = ax.plot(*empty, "+r")
    (car_outline,) = ax.plot(*empty, color=car.colour)
    (front_right_wheel,) = ax.plot(*empty, color=car.colour)
    (rear_right_wheel,) = ax.plot(*empty, color=car.colour)
    (front_left_wheel,) = ax.plot(*empty, color=car.colour)
    (rear_left_wheel,) = ax.plot(*empty, color=car.colour)
    (rear_axle,) = ax.plot(car.x, car.y, "+", color=car.colour, markersize=2)
    annotation = ax.annotate(
        f"{car.x:.1f}, {car.y:.1f}",
        xy=(car.x, car.y + 5),
        color="black",
        annotation_clip=False,
    )

    fargs = [
        Fargs(
            ax=ax,
            sim=sim,
            path=path,
            car=car,
            car_description=car_description,
            car_outline=car_outline,
            front_right_wheel=front_right_wheel,
            front_left_wheel=front_left_wheel,
            rear_right_wheel=rear_right_wheel,
            rear_left_wheel=rear_left_wheel,
            rear_axle=rear_axle,
            annotation=annotation,
            target=target,
        )
    ]

    anim = FuncAnimation(
        fig,
        animate,
        frames=sim.frames,
        init_func=lambda: None,
        fargs=fargs,
        interval=interval,
        repeat=sim.loop,
    )
    # anim.save("animation.gif", writer="imagemagick", fps=50)

    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
