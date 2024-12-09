import numpy as np
from scipy.optimize import linear_sum_assignment
import copy
import matplotlib.pyplot as plt
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

np.random.seed(1)

def find_cycle(starting_agent, prefs):
    chain = [starting_agent]
    while True:
        last_agent = chain[-1]
        top_choice = prefs[last_agent][0]

        if top_choice in chain:
            break
        else:
            chain.append(top_choice)

    cycle = [chain[-1]]
    for x in range(0, len(chain)):
        last_agent = cycle[-1]
        top_choice = prefs[last_agent][0]

        if cycle[0] == top_choice:
            break
        else:
            cycle.append(top_choice)

    assert cycle[0] == prefs[cycle[-1]][0], 'Uh oh. A formed chain doesnt actually contain a cycle'

    return cycle


def run_ttc(current_panels, waitlists):
    # Add prefixes to distinguish doctors and patients
    current_panels = {'a' + str(a): {'p' + str(p) for p in panel} for a, panel in current_panels.items()}
    waitlists = {'a' + str(a): ['p' + str(p) for p in wlist] for a, wlist in waitlists.items()}
    pop_size = sum(len(current_panels[a]) for a in current_panels)

    # To make things run faster, first limit to the set of doctors and patients that are actually going to be relevant for TTC
    # ------------------------------------
    # [i] Doctors that someone is waiting for
    current_panels_TTC = {a: panel for a, panel in current_panels.items() if a in waitlists}
    current_panels_noTTC = {a: panel for a, panel in current_panels.items() if a not in waitlists}

    # [ii] Patients that are themselves standing on a waitlist
    waiters = set(sum(waitlists.values(), []))
    current_panels_noTTC.update({a: current_panels_noTTC.get(a, set()).union(panel - waiters) for a, panel in current_panels_TTC.items()})
    current_panels_TTC = {a: panel & waiters for a, panel in current_panels_TTC.items()}

    # [iii] Doctors that have zero capacity (no patients willing to trade)
    avt_zero_capacity = {a for a, panel in current_panels_TTC.items() if len(panel) == 0}
    for a in avt_zero_capacity:
        del current_panels_TTC[a]

    # [iv] Patients waiting for zero-capacity doctors
    waiters_for_zero_cap = set(sum([waitlists[a] for a in avt_zero_capacity if a in waitlists], []))
    for a in current_panels_TTC:
        current_panels_noTTC[a] = current_panels_noTTC.get(a, set()).union(current_panels_TTC[a] & waiters_for_zero_cap)
        current_panels_TTC[a] = current_panels_TTC[a] - waiters_for_zero_cap

        # Repeat steps [iii] and [iv] until no more zero-capacity doctors
    while len(avt_zero_capacity) > 0:
        avt_zero_capacity = {a for a, panel in current_panels_TTC.items() if len(panel) == 0}
        for a in avt_zero_capacity:
            del current_panels_TTC[a]

        waiters_for_zero_cap = set(sum([waitlists[a] for a in avt_zero_capacity if a in waitlists], []))
        for a in current_panels_TTC:
            current_panels_noTTC[a] = current_panels_noTTC.get(a, set()).union(
                current_panels_TTC[a] & waiters_for_zero_cap)
            current_panels_TTC[a] = current_panels_TTC[a] - waiters_for_zero_cap

            # Verify that the total population remains the same
    total_ttc_patients = sum(len(current_panels_TTC[a]) for a in current_panels_TTC)
    total_no_ttc_patients = sum(len(current_panels_noTTC[a]) for a in current_panels_noTTC)
    if pop_size != total_ttc_patients + total_no_ttc_patients:
        print('Uh oh.. something wrong with division of people between TTC and noTTC')
        # Optionally, raise an exception or handle the error as needed

    print('{:,} people from {:,} doctors participating in TTC...'.format(total_ttc_patients, len(current_panels_TTC)))

    # Set up preference lists
    patients = set().union(*current_panels_TTC.values())
    avtales = set(current_panels_TTC.keys())

    # Capacity
    capacity = {a: len(current_panels_TTC[a]) for a in avtales}

    # Patient preferences (first prefer waitlist doctor, then current doctor)
    prefs = {}
    for a, wlist in waitlists.items():
        for p in set(wlist) & patients:
            prefs[p] = [a]
    for a, panel in current_panels_TTC.items():
        for p in panel & patients:
            if p in prefs:
                prefs[p].append(a)
            else:
                prefs[p] = [a]

    # Doctor preferences
    for a, panel in current_panels_TTC.items():
        prefs[a] = list(panel)
        prefs[a].extend([p for p in waitlists.get(a, []) if p in patients])

    # Initialize TTC panels
    ttc_panels = {a: set() for a in avtales}
    unassigned_patients = patients

    # Convert unassigned_patients to a sorted list for deterministic behavior
    unassigned_patients = sorted(unassigned_patients)

    # Run TTC algorithm
    while len(unassigned_patients) > 0:
        starting_agent = unassigned_patients[0]
        cycle = find_cycle(starting_agent, prefs)

        # Perform trades among agents in `cycle`
        for j in range(len(cycle)):
            agent = cycle[j]
            if agent in avtales:
                ttc_panels[agent].add(cycle[j - 1])
                capacity[agent] -= 1

        # Update set of unassigned patients
        unassigned_patients = [p for p in unassigned_patients if p not in cycle]

        # Update the set of available agents
        available_agents = set(unassigned_patients) | {a for a in capacity if capacity[a] > 0}

        # Delete agents with no remaining capacity from all preference lists
        prefs = {k: [x for x in v if x in available_agents] for k, v in prefs.items()}

    # Update and return current panels and waitlists
    # Merge back the two current_panels into one
    current_panels = current_panels_noTTC
    for avtale in ttc_panels:
        if avtale in current_panels:
            current_panels[avtale] = current_panels[avtale].union(ttc_panels[avtale])
        else:
            current_panels[avtale] = ttc_panels[avtale]

    # Remove anyone successfully reassigned from their waitlist
    successfully_reassigned = set()
    for a, panel in current_panels_TTC.items():
        for p in panel:
            if p not in ttc_panels.get(a, set()):
                successfully_reassigned.add(p)
    waitlists = {a: [p for p in wlist if p not in successfully_reassigned] for a, wlist in waitlists.items()}

    # Remove prefixes
    current_panels = {int(a[1:]): {int(p[1:]) for p in panel} for a, panel in current_panels.items()}
    waitlists = {int(a[1:]): [int(p[1:]) for p in wlist] for a, wlist in waitlists.items()}

    return current_panels, waitlists


# Hungarian algorithm function
def hungarian_matching(preference_lists, num_doctors, doctor_capacity, matching_dict, patients_to_match):
    num_patients_to_match = len(patients_to_match)
    unique_doctors = set(range(num_doctors))

    # Compute current load of doctors
    doctor_current_load = {doctor: 0 for doctor in unique_doctors}
    for pid, doc in matching_dict.items():
        if pid not in patients_to_match and doc is not None:
            doctor_current_load[doc] += 1

    # Compute residual capacity
    residual_capacity = {doctor: doctor_capacity - count for doctor, count in doctor_current_load.items()}

    # Filter doctors with available capacity
    available_doctors = [doctor for doctor, cap in residual_capacity.items() if cap > 0]

    # Expand doctors into available slots
    doctor_slots = []
    for doctor in available_doctors:
        for _ in range(residual_capacity[doctor]):
            doctor_slots.append(doctor)
    num_slots = len(doctor_slots)

    # Create cost matrix
    cost_matrix = np.full((num_patients_to_match, num_slots), fill_value=1e6)

    # Fill cost matrix in single-threaded manner
    for i, pid in enumerate(patients_to_match):
        pref = preference_lists[pid]
        for j, doctor in enumerate(doctor_slots):
            cost_matrix[i][j] = -pref[doctor]

    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Assign doctors
    new_matches = {}
    for i in range(len(row_ind)):
        patient_idx = patients_to_match[row_ind[i]]
        doctor_slot = col_ind[i]
        doctor_id = doctor_slots[doctor_slot]
        new_matches[patient_idx] = doctor_id

    return new_matches

# Compute total utility function
def compute_total_utility(preferences_dict, matching_dict):
    matching_array = np.full(len(preferences_dict), -1, dtype=int)
    for pid, doc in matching_dict.items():
        if doc is not None:
            matching_array[pid] = doc

    total_utility = np.sum([preferences_dict[pid][doc] for pid, doc in enumerate(matching_array) if doc != -1])
    return total_utility


# Function to plot utilities
def plot_utilities(utilities_dict, num_rounds, selected_algorithms):
    plt.figure(figsize=(12, 7))
    for algo in selected_algorithms:
        plt.plot(range(len(utilities_dict[algo])), utilities_dict[algo], label=algo.replace('_', ' ').title())
    plt.xlabel('Rounds')
    plt.ylabel('Total Utility')
    plt.title('Total Utility over Rounds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot utility function
def main():
    # Define available algorithms
    available_algorithms = ['pareto_hungarian', 'non_pareto_hungarian', 'ttc']

    # Initialize list to store selected algorithms
    selected_algorithms = []

    # Prompt user to select algorithms
    print("Select the algorithms you want to include in the simulation:")
    for algo in available_algorithms:
        while True:
            user_input = input(f"Do you want to include {algo.replace('_', ' ').title()}? (a for Yes, b for No): ").strip().lower()
            if user_input == 'a':
                selected_algorithms.append(algo)
                break
            elif user_input == 'b':
                break
            else:
                print("Invalid input. Please enter 'a' for Yes or 'b' for No.")

    if not selected_algorithms:
        print("No algorithms selected. Exiting the simulation.")
        sys.exit()

    # Initialize simulation parameters
    num_doctors = 450      # Total number of doctors
    num_patients = 300000    # Total number of patients
    doctor_capacity = 667   # Maximum patients per doctor
    num_rounds = 250        # Total number of simulation rounds

    # Initialize dictionaries to store algorithm-specific data
    patients_to_match = {algo: [] for algo in selected_algorithms}

    # Initialize patient matching histories for selected algorithms
    patient_history = {
        algo: {pid: [] for pid in range(num_patients)}
        for algo in selected_algorithms
    }

    # Initialize utility lists for selected algorithms
    utility = {algo: [] for algo in selected_algorithms}

    # Initialize sets of patients needing matching for selected algorithms
    patient_need_match = {algo: set() for algo in selected_algorithms}

    # Determine if Pareto adjustments are needed for Hungarian algorithms
    adjust_preferences = False
    if 'pareto_hungarian' in selected_algorithms:
        adjust_preferences = True

    # Initialize patient preferences
    initial_preferences_array = np.random.normal(0, 1, (num_patients, num_doctors))
    preferences_dict = {
        patient_idx: initial_preferences_array[patient_idx].tolist()
        for patient_idx in range(num_patients)
    }

    # Initialize matching dictionaries for selected algorithms
    matching_dict = {}
    for algo in selected_algorithms:
        matching_dict[algo] = {pid: None for pid in range(num_patients)}

    # Initial round: Assign patients to doctors in a round-robin fashion
    print("\n--- Round 0: Initial Assignment ---")
    all_patients = list(range(num_patients))
    initial_matches = {pid: pid // doctor_capacity for pid in all_patients}
    for algo in selected_algorithms:
        matching_dict[algo] = initial_matches.copy()
        # Initialize patient histories
        for pid in range(num_patients):
            patient_history[algo][pid].append(matching_dict[algo][pid])
        # Compute initial utility
        initial_utility = compute_total_utility(preferences_dict, matching_dict[algo])
        utility[algo].append(initial_utility)

    # Create a thread pool with as many workers as selected algorithms
    with ThreadPoolExecutor(max_workers=len(selected_algorithms)) as executor:
        # Start simulation rounds
        for round_num in range(1, num_rounds + 1):
            print(f"\n--- Round {round_num} ---")

            round_start_time = time.time()

            # Update preferences: Each patient has a 10% chance to change preferences
            changed_patients = []
            for patient_idx in range(num_patients):
                if np.random.rand() < 0.1:
                    new_preferences = np.random.normal(0, 1, num_doctors).tolist()
                    preferences_dict[patient_idx] = new_preferences
                    changed_patients.append(patient_idx)

            # Add patients who changed preferences to the set needing matching
            for algo in selected_algorithms:
                patient_need_match[algo].update(changed_patients)

            # Define a worker function for processing each algorithm
            def process_algorithm(algo):
                print(f"Processing algorithm: {algo.replace('_', ' ').title()}")
                start_time = time.time()
                patients_current = list(patient_need_match[algo])
                patients_to_match[algo].append(len(patients_current))

                if algo in ['pareto_hungarian', 'non_pareto_hungarian']:
                    # Hungarian Algorithm
                    if patients_current:
                        preference_lists = {pid: preferences_dict[pid] for pid in patients_current}

                        if algo == 'pareto_hungarian':
                            # Adjust preferences for Pareto efficiency
                            for pid in patients_current:
                                pref_list = preference_lists[pid][:]
                                current_doctor = matching_dict[algo].get(pid, None)
                                if current_doctor is not None:
                                    current_pref = pref_list[current_doctor]
                                    for doc_idx in range(len(pref_list)):
                                        if pref_list[doc_idx] < current_pref:
                                            pref_list[doc_idx] = -1e6  # Penalize less preferred doctors
                                preference_lists[pid] = pref_list

                        # Perform Hungarian Matching
                        new_matches = hungarian_matching(
                            preference_lists,
                            num_doctors,
                            doctor_capacity,
                            matching_dict[algo],
                            patients_current
                        )

                        # Update matching dictionary
                        matching_dict[algo].update(new_matches)

                        # Determine which patients have been successfully matched to a new doctor
                        patients_matched = [
                            pid for pid in patients_current
                            if new_matches.get(pid) != initial_matches.get(pid)
                        ]

                        # Remove matched patients from needing matching
                        patient_need_match[algo] -= set(patients_matched)

                        # Compute utility
                        current_utility = compute_total_utility(preferences_dict, matching_dict[algo])
                        utility[algo].append(current_utility)
                    else:
                        # If no patients need matching, utility remains the same
                        utility[algo].append(utility[algo][-1])

                    # Append current matching to patient history
                    for pid in range(num_patients):
                        patient_history[algo][pid].append(matching_dict[algo][pid])

                elif algo == 'ttc':
                    # TTC Algorithm
                    print(f"Number of patients needing TTC match before TTC: {len(patient_need_match[algo])}")
                    if patients_current:
                        # Backup old matching dictionary
                        old_matching = matching_dict[algo].copy()

                        # Build current panels and waitlists
                        current_panels = {doctor_idx: set() for doctor_idx in range(num_doctors)}
                        for patient_idx, doctor_idx in matching_dict[algo].items():
                            if doctor_idx is not None:
                                current_panels[doctor_idx].add(patient_idx)

                        waitlists = {}
                        for pid in patients_current:
                            highest_pref_doctor = np.argmax(preferences_dict[pid])
                            current_doctor = matching_dict[algo].get(pid, None)
                            if highest_pref_doctor != current_doctor:
                                if highest_pref_doctor not in waitlists:
                                    waitlists[highest_pref_doctor] = []
                                waitlists[highest_pref_doctor].append(pid)

                        # Perform TTC matching
                        updated_panels, updated_waitlists = run_ttc(current_panels, waitlists)

                        # Update matching dictionary based on updated panels
                        new_matching = {}
                        for doctor_idx, patients in updated_panels.items():
                            for pid in patients:
                                new_matching[pid] = doctor_idx

                        matching_dict[algo] = new_matching

                        # Determine which patients have been successfully matched to a new doctor
                        patients_matched = [
                            pid for pid in patients_current
                            if new_matching.get(pid) != old_matching.get(pid)
                        ]

                        # Remove matched patients from needing matching
                        patient_need_match[algo] -= set(patients_matched)

                        # Compute utility
                        current_utility = compute_total_utility(preferences_dict, matching_dict[algo])
                        utility[algo].append(current_utility)
                    else:
                        # If no patients need matching, utility remains the same
                        utility[algo].append(utility[algo][-1])

                    # Append current matching to patient history
                    for pid in range(num_patients):
                        patient_history[algo][pid].append(matching_dict[algo].get(pid, None))

                    print(f"Number of patients needing TTC match after TTC: {len(patient_need_match[algo])}")

                end_time = time.time()

            # Submit each algorithm to the thread pool
            futures = {executor.submit(process_algorithm, algo): algo for algo in selected_algorithms}

            # Wait for all threads to complete
            for future in as_completed(futures):
                algo = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f'Algorithm {algo} generated an exception: {exc}')

            round_end_time = time.time()
            # You can track total simulation time per round here if needed
            # total_simulation_time.append(round_end_time - round_start_time)

    # After simulation rounds, report average patients needing matching
    print("\n--- Simulation Results ---")
    for algo in selected_algorithms:
        avg_patients = sum(patients_to_match[algo]) / len(patients_to_match[algo]) if patients_to_match[algo] else 0
        print(f"Average patients needing match for {algo.replace('_', ' ').title()}: {avg_patients:.2f}")

    # Plot utility graph
    plot_utilities(
        utility,                # utilities_dict: dictionary containing utilities for each algorithm
        num_rounds,             # num_rounds: total number of simulation rounds
        selected_algorithms     # selected_algorithms: list of algorithms included in the simulation
    )

    print("\nSimulation completed.")

    # Return patient histories for selected algorithms
    return patient_history

# Function to track a patient's matching history
def get_patient_matching_history(patient_id, algorithm, patient_history):
    if algorithm not in patient_history:
        raise ValueError(f"Algorithm '{algorithm}' not found in patient history.")
    if patient_id not in patient_history[algorithm]:
        raise ValueError(f"Patient ID {patient_id} not found in patient history for algorithm '{algorithm}'.")
    history = patient_history[algorithm][patient_id]
    return history

patient_history = main()

'''
# Prompt user to input a patient ID for history retrieval
try:
    patient_id = int(input("\nEnter the Patient ID to view matching history (0 to 299999): ").strip())
    if not (0 <= patient_id < 300000):
        raise ValueError("Patient ID must be between 0 and 9999.")
except ValueError as ve:
    print(f"Invalid input: {ve}")
    sys.exit()


# Retrieve and print matching history for the selected patient across all selected algorithms
print("\n--- Patient Matching History ---")
for algo in patient_history:
    print(f"\nMatching history for patient {patient_id} using {algo.replace('_', ' ').title()} Algorithm:")
    history = get_patient_matching_history(patient_id, algo, patient_history)
    print(history)
'''
