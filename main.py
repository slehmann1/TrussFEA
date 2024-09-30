import math

import matplotlib.pyplot as plt
import numpy as np

INITIAL_COLOUR = "blue"
FINAL_COLOUR = "red"
SCALING_FACTOR = 0.25

# XY Coordinates
nodes = np.array([[0, 1], [0, 0], [1, 1], [1.5, 0], [2.5, 1], [3, 0]])

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
fixed_dofs = [0, 1, 4, 5]

# loads applied: (x,y)
f = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, -0.005]])


def init_stiffness_matrix(nodes, elements):
    """Creates the global K and N stiffness matrices given nodes and elements

    Args:
        nodes (np.array): (X,Y) coordinates of nodes
        elements (np.array): (Node A, Node B, EA) Specifies elements

    Returns:
        (np.array, np.array): N,K
    """
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

    return N, K


def calc_nodal_displacements(K):
    """Calculates the displacements of nodes after loads are applied

    Args:
        K (np.array): Global stiffness matrix

    Returns:
        np.array: X and Y displacements of each node after loads are applied
    """
    n = np.size(nodes, 0)
    u = np.zeros([2 * n])
    floating_nodes = np.setdiff1d(
        range(0, 2 * n), fixed_dofs
    )  # Nodes that are free to move

    # Remove fixed rows/columns
    K = np.delete(K, fixed_dofs, axis=0)
    K = np.delete(K, fixed_dofs, axis=1)

    u[floating_nodes] = np.matmul(f.flatten()[floating_nodes], np.linalg.inv(K))

    # Format u into (x,y):
    u_formatted = np.zeros([n, 2])
    u_formatted[:, 0] = u[0:-1:2]
    u_formatted[:, 1] = np.append(np.array(u[1:-2:2]), u[-1])
    return u_formatted


def print_array(arr):
    print(np.array2string(arr, 5000, formatter={"float_kind": lambda x: "%.5f" % x}))


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


def plot_nodal_displacements(pos_init, displacement, scaling_factor):
    pos_final = pos_init + displacement * scaling_factor

    # Plot Nodes
    plt.scatter(
        pos_init[:, 0],
        pos_init[:, 1],
        color=INITIAL_COLOUR,
        label="Undeformed",
    )
    plt.scatter(
        pos_final[:, 0],
        pos_final[:, 1],
        color=FINAL_COLOUR,
        label="Deformed",
    )

    # Plot Elements
    for element in elements:
        plt.plot(
            [pos_init[element[0]][0], pos_init[element[1]][0]],
            [pos_init[element[0]][1], pos_init[element[1]][1]],
            color=INITIAL_COLOUR,
        )

        plt.plot(
            [pos_final[element[0]][0], pos_final[element[1]][0]],
            [pos_final[element[0]][1], pos_final[element[1]][1]],
            color=FINAL_COLOUR,
        )

    plt.legend(loc="lower left")
    plt.title(f"Nodal Positions, Scaling Factor = {scaling_factor: .2f}")

    plt.show()


def main():
    N, K = init_stiffness_matrix(nodes, elements)
    u = calc_nodal_displacements(K)
    plot_nodal_displacements(nodes, u, SCALING_FACTOR)


if __name__ == "__main__":
    main()
