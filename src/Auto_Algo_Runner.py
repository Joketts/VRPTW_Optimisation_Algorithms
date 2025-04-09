import os
import time
import pandas as pd
from benchmark_loader import VRPBenchmarkLoader

def evaluate_time_windows(solution, customers):
    """
    computes number of time window violations and adds up lateness
    """
    violation_count = 0
    total_lateness = 0.0
    for route in solution:
        current_time = 0
        for customer in route[1:]:
            if customer == 0:
                current_time = 0
                continue
            row = customers[customers['id'] == customer]
            if row.empty:
                continue
            row = row.iloc[0]
            ready = row['ready_time']
            due = row['due_time']
            service = row['service_time']
            if current_time < ready:
                current_time = ready
            if current_time > due:
                violation_count += 1
                total_lateness += (current_time - due)
            current_time += service
    return violation_count, total_lateness

def run_benchmark_for_dataset(dataset_type, dataset_name, algorithm, generations):
    """
    loads dataset and runs algorithm chosen
    returns evaulation metrics
    """
    loader = VRPBenchmarkLoader(dataset_type=dataset_type, dataset_name=dataset_name)
    data = loader.load_data()
    customers = data["customers"]
    best_solution = None
    total_distance = None
    total_cost = None

    if algorithm.lower() == "sa":
        from SA import VRPSimulatedAnnealing
        algo = VRPSimulatedAnnealing(
            customers=customers,
            vehicle_info=data["vehicle_info"],
            initial_temp=1000,
            final_temp=1,
            alpha=0.998,
            iterations_per_temp=150
        )
        best_solution = algo.run_simulated_annealing()
        total_distance = sum(algo.calculate_route_distance(route) for route in best_solution)
        total_cost = algo.cost_function(best_solution)
    elif algorithm.lower() == "edf":
        from EDF import edf_insertion_scheduler_vrp_hybrid, compute_route_cost
        vehicle_info = data["vehicle_info"]
        num_vehicles = vehicle_info.get("num_vehicles", 1)
        best_solution = edf_insertion_scheduler_vrp_hybrid(customers, num_vehicles, candidate_list_size=7, lateness_weight=1500, parallel=True)
        total_distance = 0.0
        total_cost = 0.0
        for route in best_solution:
            d, l, c = compute_route_cost(route, customers, lateness_weight=1000)
            total_distance += d
            total_cost += c
    else:
        raise ValueError("Unsupported algorithm. Choose 'sa' or 'edf'.")

    return total_distance, total_cost, best_solution, customers

def run_benchmarks(dataset_type="solomon", algorithms=["sa", "edf"], num_runs=10, generations=500):
    """
    runs dataset against algorithm record metrics saves to file
    """
    base_path = "./datasets/solomon_100" if dataset_type.lower() == "solomon" else "./datasets/homberger_800_customer"
    # Deduplicate and sort file names
    files = sorted(set(f.lower() for f in os.listdir(base_path) if f.lower().endswith(".txt")))

    print("Files found:", files)

    results = []

    for file in files:
        dataset_name = file[:-4]
        for algorithm in algorithms:
            for run in range(1, num_runs + 1):
                print(f"Running benchmark for dataset {dataset_name} using {algorithm.upper()} algorithm, run {run}...")
                start_time = time.time()
                try:
                    total_distance, total_cost, best_solution, customers = run_benchmark_for_dataset(
                        dataset_type, dataset_name, algorithm, generations
                    )
                    run_time = time.time() - start_time
                    violation_count, total_lateness = evaluate_time_windows(best_solution, customers)

                    results.append({
                        "dataset": dataset_name,
                        "algorithm": algorithm.upper(),
                        "run": run,
                        "total_distance": total_distance,
                        "total_cost": total_cost,
                        "time_sec": run_time,
                        "violation_count": violation_count,
                        "total_lateness": total_lateness
                    })
                except Exception as e:
                    print(f"Error processing {dataset_name} with {algorithm.upper()} on run {run}: {e}")

    df = pd.DataFrame(results)
    output_file = "benchmark_results_SA_homberger_3.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    run_benchmarks(dataset_type="homberger", algorithms=["sa"], num_runs=4, generations=500)
