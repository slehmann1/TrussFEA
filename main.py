import math
import textwrap

import numpy as np

# XY Coordinates
nodes = np.array([[0, 1], [0, 0], [1, 1], [1, 0], [2, 1], [2, 0]])

# Start Node, End Node, EA
elements = np.array(
    [
        [0, 2, 1],
        [2, 4, 1],
        [0, 1, 1],
        [0, 3, 1],
        [1, 2, 1],
        [2, 3, 1],
        [2, 5, 1],
        [3, 4, 1],
        [4, 5, 1],
    ]
)

# Nodes with their positions fixed
fixed_dofs = [1, 2, 5, 6]

# loads applied: (x,y)
f = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, -0.01]])


def init_stiffness_matrix(nodes, elements):
    m = np.size(elements, 0)
    n = np.size(nodes, 0)
    K = np.zeros([2 * n, 2 * n])  # stiffness matrix
    N = np.zeros([2 * n, 4])  # nodal forces
    for i in range(0, m):
        n_1 = nodes[elements[i][0]]
        n_2 = nodes[elements[i][1]]
        length = calc_distance(n_1, n_2)
        k_el = elements[i][2] / length
        theta = calc_angle(n_1, n_2)

        # Nodal forces contribution for the element
        Ne_el = k_el * np.transpose(
            np.array(
                [-math.cos(theta), -math.sin(theta), math.cos(theta), math.sin(theta)]
            )
        )
        K_el = 1 / k_el * np.outer(Ne_el, Ne_el)

        N[i, :] = Ne_el

        # Update the appropriate elements of the overall stiffnes matrix
        K[elements[i][0] * 2][elements[i][0] * 2] += K_el[0][0]
        K[elements[i][0] * 2 + 1][elements[i][0] * 2] += K_el[1][0]

        K[elements[i][1] * 2][elements[i][0] * 2] += K_el[2][0]
        K[elements[i][1] * 2 + 1][elements[i][0] * 2] += K_el[3][0]

        K[elements[i][0] * 2][elements[i][0] * 2 + 1] += K_el[0][1]
        K[elements[i][0] * 2 + 1][elements[i][0] * 2 + 1] += K_el[1][1]

        K[elements[i][1] * 2][elements[i][0] * 2 + 1] += K_el[2][1]
        K[elements[i][1] * 2 + 1][elements[i][0] * 2 + 1] += K_el[3][1]

        K[elements[i][0] * 2][elements[i][1] * 2] += K_el[0][2]
        K[elements[i][0] * 2 + 1][elements[i][1] * 2] += K_el[1][2]

        K[elements[i][1] * 2][elements[i][1] * 2] += K_el[2][2]
        K[elements[i][1] * 2 + 1][elements[i][1] * 2] += K_el[3][2]

        K[elements[i][0] * 2][elements[i][1] * 2 + 1] += K_el[0][3]
        K[elements[i][0] * 2 + 1][elements[i][1] * 2 + 1] += K_el[1][3]

        K[elements[i][1] * 2][elements[i][1] * 2 + 1] += K_el[2][3]
        K[elements[i][1] * 2 + 1][elements[i][1] * 2 + 1] += K_el[3][3]

    print(np.array2string(K, 5000, formatter={"float_kind": lambda x: "%.5f" % x}))


def calc_distance(node_1, node_2):
    """Returns the euclidean distance between two nodes

    Args:
        node_1 (tuple): (x,y)
        node_2 (tuple): (x,y)

    Returns:
        float: Euclidean distance
    """
    return ((node_1[0] - node_2[0]) ** 2 + (node_1[1] - node_2[1]) ** 2) ** 0.5


def calc_angle(node_1, node_2):
    """Returns the angle of the line between two nodes from the +X axis

    Args:
        node_1 (tuple): (x,y)
        node_2 (tuple): (x,y)

    Returns:
        float: Angle from +X axis
    """
    return math.atan2(node_2[1] - node_1[1], node_2[0] - node_1[0])


def main():
    init_stiffness_matrix(nodes, elements)


if __name__ == "__main__":
    main()
