import matplotlib.pyplot as plt


def plot_algo_route(customers, all_routes):
    """
    plots vrp routes
    skips empty vehicles
    adds labeles
    """
    # filter out used vehicles
    used_routes = []
    for route in all_routes:
        num_customers = sum(cust != 0 for cust in route)
        if num_customers > 0:
            used_routes.append(route)

    if not used_routes:
        print("No non-empty routes found.")
        return

    plt.figure(figsize=(10, 8))
    plt.title("Vehicle Routes SA - Solomon - R106")

    # get x,y for aspect ratio
    all_x_values = []
    all_y_values = []

    color_list = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]

    # plot route in each colour
    for i, route in enumerate(used_routes):
        route_coords = []
        for cust_id in route:
            row = customers[customers['id'] == cust_id].iloc[0]
            route_coords.append((row['x'], row['y']))
            all_x_values.append(row['x'])
            all_y_values.append(row['y'])

        xs, ys = zip(*route_coords)
        color = color_list[i % len(color_list)]

        # draws routes
        for j in range(len(route_coords) - 1):
            if j == 0:
                # label first section
                plt.plot([xs[j], xs[j+1]], [ys[j], ys[j+1]],
                         color=color, marker='o',
                         label=f"Vehicle {i+1}")
            else:
                plt.plot([xs[j], xs[j+1]], [ys[j], ys[j+1]],
                         color=color, marker='o')

            # add custID , number in route
            plt.text(xs[j] + 0.3, ys[j] + 0.3,
                     f"({route[j]}, {j})",
                     fontsize=7, color='black')

        # plot final node in route
        if route_coords:
            final_j = len(route_coords) - 1
            plt.plot(xs[final_j], ys[final_j], 'o', color=color)
            plt.text(xs[final_j] + 0.3, ys[final_j] + 0.3,
                     f"({route[final_j]}, {final_j})",
                     fontsize=7, color='black')

    # make sure distances not distorted
    plt.axis('equal')

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend(loc='upper right', frameon=True)
    plt.show()
