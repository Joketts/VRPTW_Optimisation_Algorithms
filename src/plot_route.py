import matplotlib.pyplot as plt
import numpy as np

def plot_algo_route(customers, all_routes):
    """
    Plots multiple VRP routes, each with a unique colormap gradient,
    labeling each node as (customerID, visitOrder).
    The aspect ratio is kept equal for better visual clarity.

    :param customers: A pandas DataFrame with columns ['id', 'x', 'y'] for each customer/depot.
    :param all_routes: A list of routes, where each route is a list of customer IDs
                       e.g. [[0, 2, 5, 0], [0, 3, 4, 1, 0], ...]
    """
    plt.figure(figsize=(10, 8))
    plt.title("Multi-Vehicle Routes with Gradient")

    # A small set of distinct colormaps to cycle through for each vehicle route
    colormap_list = [
        'Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'Greys',
        'YlOrBr', 'YlGnBu', 'RdPu', 'BuPu'
    ]

    # Loop over each vehicle's route
    for i, route in enumerate(all_routes):
        # Gather (x, y) coordinates for this route
        route_coords = []
        for cust_id in route:
            # Find the row matching the given cust_id
            row = customers[customers['id'] == cust_id].iloc[0]
            route_coords.append((row['x'], row['y']))

        xs, ys = zip(*route_coords)

        # Create a colormap with as many distinct steps as there are points in this route
        cmap = plt.cm.get_cmap(colormap_list[i % len(colormap_list)], len(route_coords))

        # Draw each edge with a gradient color, plus a marker
        for j in range(len(route_coords) - 1):
            color = cmap(j)  # pick the color for edge j
            # For the very first edge, add a legend label so it appears in the legend
            if j == 0:
                plt.plot([xs[j], xs[j+1]], [ys[j], ys[j+1]],
                         color=color, marker='o',
                         label=f"Vehicle {i+1}")
            else:
                plt.plot([xs[j], xs[j+1]], [ys[j], ys[j+1]],
                         color=color, marker='o')

            # Label each node as (customerID, visitOrder)
            plt.text(xs[j] + 0.2, ys[j] + 0.2,
                     f"({route[j]}, {j})",
                     fontsize=8, color='black')

        # Label the final node in this route
        j = len(route_coords) - 1
        plt.text(xs[j] + 0.2, ys[j] + 0.2,
                 f"({route[j]}, {j})",
                 fontsize=8, color='black')

    # Keep the x-y aspect ratio equal so distances aren't distorted
    plt.axis('equal')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.show()
