
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from benchmark_loader import VRPBenchmarkLoader
from plot_route import plot_algo_route

def compute_route_cost(route, customers_df, lateness_weight=1000):
    """
        computes total cost of route
    """
    total_distance = 0.0
    for i in range(len(route) - 1):
        id1 = route[i]
        id2 = route[i+1]
        row1 = customers_df[customers_df['id'] == id1].iloc[0]
        row2 = customers_df[customers_df['id'] == id2].iloc[0]
        dx = row1['x'] - row2['x']
        dy = row1['y'] - row2['y']
        total_distance += math.sqrt(dx*dx + dy*dy)

    total_lateness = 0.0
    current_time = 0.0
    # skips depos
    for idx in range(1, len(route)-1):
        customer_id = route[idx]
        row = customers_df[customers_df['id'] == customer_id].iloc[0]
        ready = row['ready_time']
        due = row['due_time']
        service = row['service_time']
        if current_time < ready:
            current_time = ready
        finish_time = current_time + service
        lateness = max(0, finish_time - due)
        total_lateness += lateness
        current_time = finish_time
    total_cost = total_distance + lateness_weight * total_lateness
    return total_distance, total_lateness, total_cost

def candidate_insertion_delta(route, pos, candidate, customers_df, lateness_weight=1000):
    """
        gets the change in cost when inserting new customer
    """
    new_route = route[:pos] + [candidate] + route[pos:]
    old_cost = compute_route_cost(route, customers_df, lateness_weight)[2]
    new_cost = compute_route_cost(new_route, customers_df, lateness_weight)[2]
    delta = new_cost - old_cost
    return delta, new_route

def evaluate_insertion(route, pos, candidate, v_id, customers_df, lateness_weight):
    """
    evauluates inserting customer into route at any given position
    """
    delta, new_route = candidate_insertion_delta(route, pos, candidate, customers_df, lateness_weight)
    return v_id, pos, candidate, (delta, new_route)

def edf_insertion_scheduler_vrp_hybrid(customers_df, num_vehicles, candidate_list_size=10, lateness_weight=1000, parallel=True):

    unscheduled = set(customers_df[customers_df['id'] != 0]['id'].tolist())

    # Initialize each vehicle route with depo at start n end
    routes = {v: [0, 0] for v in range(1, num_vehicles+1)}

    while unscheduled:
        # shorten list by EDF
        candidate_list = sorted(list(unscheduled), key=lambda c: customers_df[customers_df['id'] == c].iloc[0]['due_time'])
        candidate_list = candidate_list[:candidate_list_size]

        best_delta = float('inf')
        best_insertion = None

        if parallel:
            tasks = []
            with ProcessPoolExecutor() as executor:
                # for each customer submit tasks for every possible insertion in all routes
                for candidate in candidate_list:
                    for v, route in routes.items():
                        for pos in range(1, len(route)):  # valid positions for insertion between nodes
                            tasks.append(executor.submit(
                                evaluate_insertion,
                                route, pos, candidate, v, customers_df, lateness_weight
                            ))
                for future in as_completed(tasks):
                    try:
                        v_id, pos, cand, (delta, new_route) = future.result()
                        if delta < best_delta:
                            best_delta = delta
                            best_insertion = (cand, v_id, pos, new_route)
                    except Exception as e:
                        print(f"Error in candidate evaluation: {e}")
        else:
            for candidate in candidate_list:
                for v, route in routes.items():
                    for pos in range(1, len(route)):
                        delta, new_route = candidate_insertion_delta(route, pos, candidate, customers_df, lateness_weight)
                        if delta < best_delta:
                            best_delta = delta
                            best_insertion = (candidate, v, pos, new_route)

        if best_insertion is None:
            break
        candidate, vehicle_id, pos, new_route = best_insertion
        routes[vehicle_id] = new_route
        unscheduled.remove(candidate)
        print(f"Inserted customer {candidate} into vehicle {vehicle_id} at position {pos}, cost increase: {best_delta:.2f}")

    return list(routes.values())

if __name__ == '__main__':
    solomon_loader = VRPBenchmarkLoader(dataset_type="solomon", dataset_name="r101")
    solomon_data = solomon_loader.load_data()
    customers_df = solomon_data["customers"]
    vehicle_info = solomon_data["vehicle_info"]
    num_vehicles = vehicle_info.get("num_vehicles", 1)

    routes = edf_insertion_scheduler_vrp_hybrid(customers_df, num_vehicles, candidate_list_size=5, lateness_weight=1000, parallel=True)

    total_distance = 0.0
    print("\nHybrid EDF Insertion Routes for VRP Customers:")
    for idx, route in enumerate(routes, start=1):
        d, l, cost = compute_route_cost(route, customers_df, lateness_weight=1000)
        total_distance += d
        print(f"Vehicle {idx}: {route} | Distance: {d:.2f}, Total Lateness: {l:.2f}, Route Cost: {cost:.2f}")

    print(f"\nTotal Distance across all vehicles: {total_distance:.2f}")

    plot_algo_route(customers_df, routes)
