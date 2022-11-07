import random
import argparse
import os

# WANT: generate URDF

# Use all primitives available in URDF: box (cube), cylinder, sphere
# need directory, number of objects to generate, maximum bounding box size?
# do we need to generate mass?

# probably a good idea to copy YCB/Shapenet structure so that our list gens
# should do less work

# directory structure is /SOURCE_ROOT/urdf (there are no visual models since we're using primitives, so that's it)
# urdf filename structure is ObjectName_SomethingThatLooksHashlike.urdf

# sample file:

PRIMITIVES = ['cylinder', 'box', 'sphere']
URDF_TEMPLATE = '''<?xml version="1.0"?>
<robot  name="UNNAMED_%i" >
 <link  name="UNNAMED_%i" >
  <inertial >
   <origin  rpy="0 0 0"  xyz="0 0 0" />
   <mass  value="1.0" />
   <inertia  ixx="0.001"  izz="0.001"  iyy="0.001" />
  </inertial>
  <visual >
   <geometry >
     <%s />
   </geometry>
  </visual>
  <collision >
   <geometry >
     <%s />
   </geometry>
  </collision>
 </link>
</robot>'''


def main(args):
    for p in args.primitives:
        if p not in PRIMITIVES:
            print('Please ensure all primitives specified are valid. Valid options: cylinder, cube, sphere.')
            return

    # make directories
    data_dir = (args.directory if args.directory[-1] == '/' else args.directory + '/') + 'urdfs/'
    try:
        os.mkdir(args.directory)  # do this first to ensure there isn't an existing directory with this name
        os.mkdir(data_dir)
    except FileExistsError as e:
        print('Directory already exists.')
        return
    except FileNotFoundError as e:
        print('Invalid directory.')
        return

    if 'cylinder' in args.primitives:
        for i in range(args.n_prim):
            min_len, min_rad = args.cylinder_min_len_rad
            max_len, max_rad = args.cylinder_max_len_rad

            side_len = random.uniform(min_len, max_len)
            rad = random.uniform(min_rad, max_rad)

            urdf_text = URDF_TEMPLATE % (2 * i,
                                         2 * i + 1,
                                         'cylinder length=\"%f\" radius=\"%f\"' % (side_len, rad),
                                         'cylinder length=\"%f\" radius=\"%f\"' % (side_len, rad))
            f = open(data_dir + 'Cylinder_%i.urdf' % hash(random.uniform(0, 1)), 'w')
            f.write(urdf_text)
            f.close()

    if 'box' in args.primitives:
        for i in range(args.n_prim):
            side_min1, side_min2, side_min3 = args.box_min_size
            side_max1, side_max2, side_max3 = args.box_max_size

            side_len1 = random.uniform(side_min1, side_max1)
            side_len2 = random.uniform(side_min2, side_max2)
            side_len3 = random.uniform(side_min3, side_max3)

            urdf_text = URDF_TEMPLATE % (2 * i,
                                         2 * i + 1,
                                         'box size=\"%f %f %f\"' % (side_len1, side_len2, side_len3),
                                         'box size=\"%f %f %f\"' % (side_len1, side_len2, side_len3))
            f = open(data_dir + 'Box_%i.urdf' % hash(random.uniform(0, 1)), 'w')
            f.write(urdf_text)
            f.close()

    if 'sphere' in args.primitives:
        for i in range(args.n_prim):
            rad = random.uniform(args.circle_min_rad, args.circle_max_rad)
            urdf_text = URDF_TEMPLATE % (2 * i,
                                         2 * i + 1,
                                         'sphere radius=\"%f\"' % (rad),
                                         'sphere radius=\"%f\"' % (rad))
            f = open(data_dir + 'Sphere_%i.urdf' % hash(random.uniform(0, 1)), 'w')
            f.write(urdf_text)
            f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', type=str, required=True,
                        help='path and name of directory to store primitive data')
    parser.add_argument('--primitives', '-p', type=str, nargs='+', required=True,
                        help='primitives to generate in set. options are: cylinder, box, and sphere.')
    parser.add_argument('--n_prim', '-n', type=int, required=True,
                        help='number of objects to generate (this is done per primitive)')
    parser.add_argument('--circle-min-rad', type=float, default=0.01)
    parser.add_argument('--circle-max-rad', type=float, default=0.03)
    parser.add_argument('--box-min-size', type=float, nargs=3, default=[0.01, 0.01, 0.01])
    parser.add_argument('--box-max-size', type=float, nargs=3, default=[0.07, 0.07, 0.07])
    parser.add_argument('--cylinder-min-len-rad', type=float, nargs=2, default=[0.01, 0.01])
    parser.add_argument('--cylinder-max-len-rad', type=float, nargs=2, default=[0.07, 0.03])

    args = parser.parse_args()
    main(args)
