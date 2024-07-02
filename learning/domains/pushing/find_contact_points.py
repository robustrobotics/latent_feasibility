import pybullet as p
import pybullet_data
import math
import time
import argparse


def find_contact_point_and_check_push(
    object_name,
    angle_degrees,
    push_velocity,
    object_mass,
    object_friction,
    com_shift,
    translation,
    gui=False,
    logging=False,
):
    """
    Run the simulation given a specific amount of parameters.
    Returns the tuple with the contact point, whether the push was "successful", and the trajectory.
    A push being successful is based on whether the object and the push were close in distance.
    """

    angle_radians = math.radians(angle_degrees)

    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane_id = p.loadURDF("plane.urdf")
    cube_id = p.loadURDF(object_name, [0, 0, 1], globalScaling=1)
    sphere_id = p.loadURDF(
        "sphere_small.urdf",
        [2 * math.cos(angle_radians), 2 * math.sin(angle_radians), 1],
        globalScaling=1,
    )

    p.setGravity(0, 0, -9.81)

    p.changeDynamics(cube_id, -1, mass=object_mass, lateralFriction=object_friction)
    # p.changeDynamics(sphere_id, -1, mass=object_mass, lateralFriction=object_friction)

    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
    cube_pos = [
        cube_pos[0] + com_shift[0] + translation[0],
        cube_pos[1] + com_shift[1] + translation[1],
        cube_pos[2] + com_shift[2] + translation[2],
    ]
    p.resetBasePositionAndOrientation(cube_id, cube_pos, cube_orn)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

    initial_distance = calculate_distance(cube_id, sphere_id)
    # print(f"Initial Distance: {initial_distance}")

    force_magnitude = 1
    force_x = -force_magnitude * math.cos(angle_radians)
    force_y = -force_magnitude * math.sin(angle_radians)

    SETTLING_STEPS = 1000
    SIMULATION_STEPS = 5000
    SUCCESS_DISTANCE_THRESHOLD = 0.5  # Maximum distance to consider push successful

    for _ in range(SETTLING_STEPS):
        p.stepSimulation()

    contact_points = []
    positions = []
    for step in range(SIMULATION_STEPS):
        p.resetBaseVelocity(
            sphere_id,
            linearVelocity=[push_velocity * force_x, push_velocity * force_y, 0],
        )
        p.stepSimulation()

        if logging and step % 100 == 0:
            cube_pos, cube_orientation = p.getBasePositionAndOrientation(cube_id)
            cube_both = []
            for x in cube_pos:
                cube_both.append(x)
            for x in cube_orientation:
                cube_both.append(x)
            positions.append(tuple(cube_both))

        step_contacts = p.getContactPoints(bodyA=cube_id, bodyB=sphere_id)
        if step_contacts:
            contact_points.extend(step_contacts)

        if gui:
            time.sleep(0.0001)

    final_distance = calculate_distance(cube_id, sphere_id)
    # print(f"Final Distance: {final_distance}")

    p.disconnect()

    push_successful = final_distance <= SUCCESS_DISTANCE_THRESHOLD

    if contact_points:
        # Average contact point
        avg_contact_point = [
            sum(c[5][i] for c in contact_points) / len(contact_points) for i in range(3)
        ]
        return avg_contact_point, push_successful, positions if logging else None
    else:
        return None, push_successful, positions if logging else None


def calculate_distance(bodyA, bodyB):
    posA, _ = p.getBasePositionAndOrientation(bodyA)
    posB, _ = p.getBasePositionAndOrientation(bodyB)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(posA, posB)))


def main():
    parser = argparse.ArgumentParser(
        description="Simulate object pushing and evaluate success."
    )
    parser.add_argument(
        "--angle", type=float, default=90, help="Angle of push in degrees"
    )
    parser.add_argument(
        "--mass", type=float, default=0.00001, help="Mass of the objects"
    )
    parser.add_argument(
        "--friction", type=float, default=0.5, help="Friction of the objects"
    )
    parser.add_argument("--velocity", type=float, default=2, help="Push velocity")
    parser.add_argument("--gui", action="store_true", help="Use GUI for visualization")
    parser.add_argument("--logging", action="store_true", help="Log object positions")
    args = parser.parse_args()

    translation = [0, 0, 0]
    com_shift = [0, 0, 0]

    try:
        contact_point, push_successful, positions = find_contact_point_and_check_push(
            "Primitive::Box_3161924807452160_0.188m_0.274f_-0.012cx_0.035cy_-0.011cz.urdf",
            args.angle,
            args.velocity,
            args.mass,
            args.friction,
            com_shift,
            translation,
            args.gui,
            args.logging,
        )

        if contact_point:
            print(f"Contact Point at {args.angle} degrees: {contact_point}")
        else:
            print(f"No contact point found at {args.angle} degrees")

        print(
            "The push was successful."
            if push_successful
            else "The push was not successful."
        )

        if args.logging and positions:
            print("Object positions logged:")
            for pos in positions:
                print(pos)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
