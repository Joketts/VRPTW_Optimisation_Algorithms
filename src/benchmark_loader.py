import os
import pandas as pd


class VRPBenchmarkLoader:
    def __init__(self, dataset_type, dataset_name):
        """
        initialize dataset loader
        """
        self.dataset_type = dataset_type.lower()
        self.dataset_name = dataset_name.lower()
        self.file_path = self._get_file_path()

    def _get_file_path(self):
        if self.dataset_type == "solomon":
            base_path = "./datasets/solomon_100"
        else:
            base_path = "./datasets/homberger_800_customer"

        available_files = {f.lower(): f for f in os.listdir(base_path)}

        file_key = self.dataset_name + ".txt"
        if self.dataset_name + ".txt" in available_files:
            return f"{base_path}/{available_files[self.dataset_name + '.txt']}"
        else:
            raise FileNotFoundError(f"Dataset file '{self.dataset_name}.txt' not found in {base_path}")

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        with open(self.file_path, "r") as file:
            lines = file.readlines()

        vehicle_info = {}
        customers = []
        reading_nodes = False

        for line in lines:
            line = line.strip()
            if not line or "VEHICLE" in line:
                continue

            tokens = line.split()
            if len(tokens) == 2 and "CAPACITY" in vehicle_info:
                continue

            if len(tokens) == 2 and tokens[0].isdigit():
                vehicle_info["num_vehicles"], vehicle_info["capacity"] = map(int, tokens)
                continue

            if "CUST NO." in line:
                reading_nodes = True
                continue

            if reading_nodes:
                cust_id = int(tokens[0])
                x, y = float(tokens[1]), float(tokens[2])
                demand = int(tokens[3])
                ready_time = int(tokens[4])
                due_time = int(tokens[5])
                service_time = int(tokens[6])

                customers.append({
                    "id": cust_id,
                    "x": x,
                    "y": y,
                    "demand": demand,
                    "ready_time": ready_time,
                    "due_time": due_time,
                    "service_time": service_time
                })

        return {
            "vehicle_info": vehicle_info,
            "customers": pd.DataFrame(customers)
        }
