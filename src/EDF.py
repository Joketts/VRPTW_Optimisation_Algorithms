#!/usr/bin/env python3
"""
edf_insertion_scheduler_vrp_hybrid.py

Hybrid EDF–Insertion Scheduler with Parallelization:
  - Restricts unscheduled customers (by earliest due time) to a candidate list.
  - For each candidate, evaluates every possible insertion position in every route using incremental cost calculation.
  - Parallelizes these evaluations using ProcessPoolExecutor.

The cost for a route is defined as its total Euclidean distance plus a penalty (weighted by lateness_weight) for any lateness.
"""

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from benchmark_loader import VRPBenchmarkLoader
from plot_route import plot_algo_route

def compute_route_cost(route, customers_df, lateness_weight=1000):
    """
    Computes the total cost of a route (distance plus lateness penalty).

    Args:
        route (list): List of customer ids (with depot at start and end).
        customers_df (DataFrame): Contains columns 'x', 'y', 'ready_time', 'due_time', 'service_time'.
        lateness_weight (float): Weight factor applied to total lateness.

    Returns:
        tuple: (total_distance, total_lateness, total_cost)
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
    # Skip the first depot and the final depot
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
    Computes the incremental change in cost (delta) when inserting a candidate customer
    into the given route at the specified position.

    Args:
        route (list): Current route (list of customer ids).
        pos (int): Insertion position (between route[pos-1] and route[pos]).
        candidate: Candidate customer id.
        customers_df (DataFrame): Customer data.
        lateness_weight (float): Weight factor for lateness.

    Returns:
        tuple: (delta_cost, new_route) where delta_cost is the increase in cost.
    """
    new_route = route[:pos] + [candidate] + route[pos:]
    old_cost = compute_route_cost(route, customers_df, lateness_weight)[2]
    new_cost = compute_route_cost(new_route, customers_df, lateness_weight)[2]
    delta = new_cost - old_cost
    return delta, new_route

# Define a top-level function for evaluation (instead of a lambda)
def evaluate_insertion(route, pos, candidate, v_id, customers_df, lateness_weight):
    """
    Evaluates inserting a candidate into a route at the given position.

    Returns:
        tuple: (v_id, pos, candidate, (delta, new_route))
    """
    delta, new_route = candidate_insertion_delta(route, pos, candidate, customers_df, lateness_weight)
    return v_id, pos, candidate, (delta, new_route)

def edf_insertion_scheduler_vrp_hybrid(customers_df, num_vehicles, candidate_list_size=10, lateness_weight=1000, parallel=True):
    """
    Hybrid EDF Insertion Scheduler:
      - Restricts unscheduled candidates to the top 'candidate_list_size' by earliest due date.
      - Evaluates all insertion positions (using incremental cost calculation) for each candidate.
      - Optionally parallelizes candidate evaluations.

    Args:
        customers_df (DataFrame): Contains customer info.
        num_vehicles (int): Number of available vehicles.
        candidate_list_size (int): Maximum number of unscheduled candidates to consider at each iteration.
        lateness_weight (float): Weight for lateness in the cost function.
        parallel (bool): Whether to parallelize candidate evaluations.

    Returns:
        list: A list of routes (each route is a list of customer ids with depot at start and end).
    """
    unscheduled = set(customers_df[customers_df['id'] != 0]['id'].tolist())
    # Initialize each vehicle's route with depot at start and end.
    routes = {v: [0, 0] for v in range(1, num_vehicles+1)}

    while unscheduled:
        # Restrict candidate list by EDF (earliest due date)
        candidate_list = sorted(list(unscheduled), key=lambda c: customers_df[customers_df['id'] == c].iloc[0]['due_time'])
        candidate_list = candidate_list[:candidate_list_size]

        best_delta = float('inf')
        best_insertion = None  # (candidate, vehicle_id, position, new_route)

        if parallel:
            tasks = []
            with ProcessPoolExecutor() as executor:
                # Submit tasks for each candidate and every possible insertion in every route.
                for candidate in candidate_list:
                    for v, route in routes.items():
                        for pos in range(1, len(route)):  # valid insertion positions between nodes
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
            # Sequential evaluation
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
    # Load the Solomon dataset (e.g., c102)
    solomon_loader = VRPBenchmarkLoader(dataset_type="homberger", dataset_name="C1_8_1")
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

    # Visualize the routes
    plot_algo_route(customers_df, routes)
