#!/usr/bin/env python3
import os
import time
import math
import pandas as pd
from benchmark_loader import VRPBenchmarkLoader
from EDF import edf_insertion_scheduler_vrp_hybrid, compute_route_cost

def evaluate_time_windows(solution, customers_df):
    """
    Evaluates time window performance for a given solution.
    Returns the number of violations and the total lateness.
    """
    violation_count = 0
    total_lateness = 0.0
    for route in solution:
        current_time = 0.0
        # Ignore depot at start and end
        for customer in route[1:-1]:
            row = customers_df[customers_df['id'] == customer].iloc[0]
            ready = row['ready_time']
            due = row['due_time']
            service = row['service_time']
            if current_time < ready:
                current_time = ready
            finish_time = current_time + service
            lateness = max(0, finish_time - due)
            if lateness > 0:
                violation_count += 1
            total_lateness += lateness
            current_time = finish_time
    return violation_count, total_lateness

def run_edf_and_save(dataset_name, candidate_list_size=10, lateness_weight=1000):
    """
    Runs the hybrid EDF algorithm on the specified homberger dataset,
    computes metrics, and appends the results to a CSV file.
    """
    # Load dataset using VRPBenchmarkLoader (dataset type is homberger)
    loader = VRPBenchmarkLoader("homberger", dataset_name)
    data = loader.load_data()
    customers_df = data["customers"]
    vehicle_info = data["vehicle_info"]
    num_vehicles = vehicle_info.get("num_vehicles", 1)

    start_time = time.time()
    solution = edf_insertion_scheduler_vrp_hybrid(customers_df, num_vehicles, candidate_list_size, lateness_weight, parallel=True)
    time_sec = time.time() - start_time

    total_distance = 0.0
    total_cost = 0.0
    for route in solution:
        d, l, cost = compute_route_cost(route, customers_df, lateness_weight)
        total_distance += d
        total_cost += cost

    violation_count, total_lateness = evaluate_time_windows(solution, customers_df)

    # Create a result record with the desired columns.
    result = {
        "dataset": "homberger",
        "algorithm": "EDF",
        "run": "run 1",
        "total_distance": total_distance,
        "total_cost": total_cost,
        "time_sec": time_sec,
        "violation_count": violation_count,
        "total_lateness": total_lateness
    }

    output_file = "benchmark_results_EDF_homberger.csv"
    df_result = pd.DataFrame([result])
    # Append new results if the file already exists.
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df_combined = pd.concat([df_existing, df_result], ignore_index=True)
        df_combined.to_csv(output_file, index=False)
    else:
        df_result.to_csv(output_file, index=False)
    print("Results saved to", output_file)

if __name__ == "__main__":
    # Specify the dataset name (for example, "RC2_8_10" for a homberger dataset)
    dataset_name = "C1_8_3"
    run_edf_and_save(dataset_name)
