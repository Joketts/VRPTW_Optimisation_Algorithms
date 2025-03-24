# Adjusting file path handling for Solomon dataset
import os
import pandas as pd


class VRPBenchmarkLoader:
    def __init__(self, dataset_type, dataset_name):
        """
        Initialize dataset loader for Solomon or Homberger datasets.

        :param dataset_type: 'solomon' or 'homberger'
        :param dataset_name: specific dataset file (e.g., 'C101', 'R1_8_1')
        """
        self.dataset_type = dataset_type.lower()
        self.dataset_name = dataset_name.lower()
        self.file_path = self._get_file_path()

    def _get_file_path(self):
        """Determines the correct file path based on dataset type and name."""
        if self.dataset_type == "solomon":
            base_path = "./datasets/solomon_100"  # Normal path as a string
        else:
            base_path = "./datasets/homberger_800_customer"

        available_files = {f.lower(): f for f in os.listdir(base_path)}

        file_key = self.dataset_name + ".txt"
        if self.dataset_name + ".txt" in available_files:
            return f"{base_path}/{available_files[self.dataset_name + '.txt']}"  # Use normal path
        else:
            raise FileNotFoundError(f"Dataset file '{self.dataset_name}.txt' not found in {base_path}")

    def load_data(self):
        """
        Parses the dataset file and returns structured data.

        :return: dict containing vehicles, customers, and problem details.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        with open(self.file_path, "r") as file:
            lines = file.readlines()

        # Initialize data structures
        vehicle_info = {}
        customers = []
        reading_nodes = False

        for line in lines:
            line = line.strip()
            if not line or "VEHICLE" in line:
                continue  # Skip empty lines and section headers

            tokens = line.split()
            if len(tokens) == 2 and "CAPACITY" in vehicle_info:
                continue

            # Vehicle section
            if len(tokens) == 2 and tokens[0].isdigit():
                vehicle_info["num_vehicles"], vehicle_info["capacity"] = map(int, tokens)
                continue

            # Customer section header
            if "CUST NO." in line:
                reading_nodes = True
                continue

            # Read customer node information
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
            "customers": pd.DataFrame(customers)  # Return as Pandas DataFrame for easier handling
        }


# Retry loading the datasets
# solomon_loader = VRPBenchmarkLoader(dataset_type="solomon", dataset_name="c101")
# solomon_data = solomon_loader.load_data()

# homberger_loader = VRPBenchmarkLoader(dataset_type="homberger", dataset_name="RC2_8_10")
# homberger_data = homberger_loader.load_data()

# Display results
# print("Vehicle Info:", solomon_data["vehicle_info"])
# print("First 5 Customers:\n", solomon_data["customers"].head())
