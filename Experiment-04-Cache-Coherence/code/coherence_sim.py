
import sys


class CacheSimulator:
    def __init__(self, protocol):
        self.protocol = protocol
        self.caches = ["I", "I"]
        self.bus_transactions = 0
        self.writebacks = 0
        self.cache_to_cache = 0
        self.silent_upgrades = 0

    def read(self, core):
        other = 1 - core

        if self.caches[core] != "I":
            return

        other_state = self.caches[other]
        self.bus_transactions += 1

        if other_state == "M":
            self.cache_to_cache += 1

            if self.protocol == "MOESI":
                self.caches[other] = "O"
            else:
                self.writebacks += 1
                self.caches[other] = "S"

            self.caches[core] = "S"

        elif other_state == "E":
            self.cache_to_cache += 1
            self.caches[other] = "S"
            self.caches[core] = "S"

        elif other_state == "O":
            self.cache_to_cache += 1
            self.caches[core] = "S"

        elif other_state == "S":
            self.caches[core] = "S"

        else:
            self.caches[core] = (
                "E" if self.protocol in ("MESI", "MOESI") else "S"
            )

    def write(self, core):
        other = 1 - core
        state = self.caches[core]

        if state == "M":
            return

        if state == "E":
            self.caches[core] = "M"
            self.silent_upgrades += 1
            return

        if state in ("S", "O"):
            self.bus_transactions += 1
            self.caches[other] = "I"
            self.caches[core] = "M"
            return

        self.bus_transactions += 1

        if self.caches[other] == "M":
            self.writebacks += 1

        self.caches[other] = "I"
        self.caches[core] = "M"

    def run(self, operations):
        for core, operation in operations:
            if operation == "R":
                self.read(core)
            else:
                self.write(core)

    def report(self):
        return (
            self.bus_transactions,
            self.writebacks,
            self.cache_to_cache,
            self.silent_upgrades
        )


def make_operations(count):
    operations = []

    for _ in range(count):
        operations.extend([
            (0, "R"),
            (0, "W"),
            (1, "R"),
            (1, "W")
        ])

    return operations


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if count <= 0:
        print("Please enter a positive number of iterations.")
        return

    operations = make_operations(count)

    print("CACHE COHERENCE PROTOCOL SIMULATOR")
    print("Two-core read/write workload")
    print("Iterations:", count)

    print("\nProtocol Comparison")
    print("-" * 78)

    print(
        f"{'Protocol':<12}"
        f"{'Bus Transactions':<20}"
        f"{'Writebacks':<14}"
        f"{'C2C Transfers':<17}"
        f"{'Silent Upgrades'}"
    )

    for protocol in ("MSI", "MESI", "MOESI"):
        simulator = CacheSimulator(protocol)
        simulator.run(operations)

        bus, writebacks, c2c, silent = simulator.report()

        print(
            f"{protocol:<12}"
            f"{bus:<20}"
            f"{writebacks:<14}"
            f"{c2c:<17}"
            f"{silent}"
        )

    print("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()
