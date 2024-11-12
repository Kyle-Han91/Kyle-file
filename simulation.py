import numpy as np
from scipy.optimize import linear_sum_assignment
import copy
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor

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

    # Proceed with the rest of the TTC algorithm as before
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
    if not available_doctors:
        raise Exception("No available doctors to assign.")

    # Expand doctors into available slots
    doctor_slots = []
    for doctor in available_doctors:
        for _ in range(residual_capacity[doctor]):
            doctor_slots.append(doctor)
    num_slots = len(doctor_slots)
    if num_slots < num_patients_to_match:
        raise Exception("Not enough available slots to assign all patients.")

    # Create cost matrix
    cost_matrix = np.full((num_patients_to_match, num_slots), fill_value=1e6)

    def fill_cost_row(i, pid):
        row_costs = np.full(num_slots, 1e6)
        pref = preference_lists[pid]
        for j, doctor in enumerate(doctor_slots):
            row_costs[j] = -pref[doctor]
        return i, row_costs

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(fill_cost_row, i, pid) for i, pid in enumerate(patients_to_match)]
        for future in futures:
            i, row_costs = future.result()
            cost_matrix[i] = row_costs

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


# Plot utility function
def plot_utilities(utility_hungarian, utility_ttc, utility_no_matching, num_rounds):
    rounds = range(0, num_rounds + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, utility_hungarian, label='Hungarian Algorithm', color='blue')
    plt.plot(rounds, utility_ttc, label='TTC Algorithm', color='red')
    plt.plot(rounds, utility_no_matching, label='No Matching', color='green', linestyle='--')
    plt.xlabel('Round Number')
    plt.ylabel('Total Utility')
    plt.title('Total Utility Over Rounds for Hungarian, TTC Algorithms, and No Matching')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_execution_times(time_hungarian, time_ttc, num_rounds,total_simulation_time):
    rounds = range(1, num_rounds + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, time_hungarian, label='Hungarian Algorithm', color='blue')
    plt.plot(rounds, time_ttc, label='TTC Algorithm', color='red')
    plt.plot(rounds, total_simulation_time, label='Total', color='green')
    plt.xlabel('Round Number')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time per Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    num_doctors = 450        # Total number of doctors
    num_patients = 300000      # Total number of patients
    doctor_capacity = 667   # Maximum patients per doctor
    num_rounds = 500        # Total number of simulation rounds
    time_hungarian = []
    time_ttc = []
    total_simulation_time = []

    hungarian_0_time_start = time.time()
    # Initialize patient preferences
    initial_preferences_array = np.random.normal(0, 1, (num_patients, num_doctors))
    preferences_dict = {
        patient_idx: initial_preferences_array[patient_idx].tolist()
        for patient_idx in range(num_patients)
    }

    # Initialize matching dictionaries
    matching_dict_initial = {pid: None for pid in range(num_patients)}

    # Initialize patient matching history dictionaries
    patient_history_hungarian = {pid: [] for pid in range(num_patients)}
    patient_history_ttc = {pid: [] for pid in range(num_patients)}
    patient_history_no_op = {pid: [] for pid in range(num_patients)}

    # Initial round: use Hungarian algorithm for initial assignment
    print("--- Round 0: Initial Assignment using Hungarian Algorithm ---")
    all_patients = list(range(num_patients))
    preference_lists_initial = {pid: preferences_dict[pid] for pid in all_patients}

    for doctor_idx in range(num_doctors):
        for patient_idx in range(num_patients):
            matching_dict_initial[patient_idx] = patient_idx // doctor_capacity

    new_matches_initial = matching_dict_initial

    # Update matching dictionary with new matches
    matching_dict_initial.update(new_matches_initial)

    # Record initial utility
    initial_utility = compute_total_utility(preferences_dict, matching_dict_initial)

    # Copy three matching dictionaries
    matching_dict_hungarian = matching_dict_initial.copy()
    matching_dict_ttc = matching_dict_initial.copy()
    matching_dict_no_op = matching_dict_initial.copy()

    # Append initial matching to patient histories
    for pid in range(num_patients):
        patient_history_hungarian[pid].append(matching_dict_hungarian[pid])
        patient_history_ttc[pid].append(matching_dict_ttc[pid])
        patient_history_no_op[pid].append(matching_dict_no_op[pid])

    # Initialize utility lists
    utility_hungarian = [initial_utility]
    utility_ttc = [initial_utility]
    utility_no_matching = [initial_utility]

    # Initialize sets of patients needing matching
    patient_need_match_hungarian = set()
    patient_need_match_ttc = set()
    patient_need_match_no_op = set()

    hungarian_0_time_end = time.time()
    hungarian_0_time = hungarian_0_time_end - hungarian_0_time_start
    print(hungarian_0_time)

    # Start Running
    for round_num in range(1, num_rounds + 1):
        print(f"--- Round {round_num} ---")

        total_start_time = time.time()

        # Update preferences
        changed_patients = []
        for patient_idx in range(num_patients):
            if np.random.rand() < 0.1:
                new_preferences = np.random.normal(0, 1, num_doctors).tolist()
                preferences_dict[patient_idx] = new_preferences
                changed_patients.append(patient_idx)

        # Add patients who changed preferences to the set needing matching
        for pid in changed_patients:
            patient_need_match_hungarian.add(pid)
            patient_need_match_ttc.add(pid)
            patient_need_match_no_op.add(pid)

        # Group 1: Hungarian algorithm
        patients_to_match_hungarian = list(patient_need_match_hungarian)
        if patients_to_match_hungarian:
            preference_lists_hungarian = {pid: preferences_dict[pid] for pid in patients_to_match_hungarian}

            start_time_hungarian = time.time()
            new_matches_hungarian = hungarian_matching(
                    preference_lists_hungarian,
                    num_doctors,
                    doctor_capacity,
                    matching_dict_hungarian,
                    patients_to_match_hungarian
            )
            end_time_hungarian = time.time()
            time_hungarian.append(end_time_hungarian - start_time_hungarian)


            # Update matching dictionary
            matching_dict_hungarian.update(new_matches_hungarian)

            # Check which patients' matches have changed
            patients_matched = []
            for pid in patients_to_match_hungarian:
                if new_matches_hungarian[pid] != matching_dict_initial[pid]:
                    patients_matched.append(pid)
            # Remove patients who have been matched to a new doctor
            patient_need_match_hungarian -= set(patients_matched)

            # Compute utility
            utility_h = compute_total_utility(preferences_dict, matching_dict_hungarian)
            utility_hungarian.append(utility_h)
        else:
            # If no patients need matching, utility remains the same
            utility_hungarian.append(utility_hungarian[-1])

        # Append current matching to patient history
        for pid in range(num_patients):
            patient_history_hungarian[pid].append(matching_dict_hungarian[pid])

        patients_to_match_ttc = list(patient_need_match_ttc)
        print(f"Number of patients needing TTC match before TTC: {len(patient_need_match_ttc)}")
        if patients_to_match_ttc:

            start_time_ttc = time.time()

            old_matching_dict_ttc = matching_dict_ttc.copy()

            # Build current panels and waitlists
            current_panels = {}
            waitlists = {}
            for doctor_idx in range(num_doctors):
                current_panels[doctor_idx] = set()
            for patient_idx, doctor_idx in matching_dict_ttc.items():
                if doctor_idx is not None:
                    current_panels[doctor_idx].add(patient_idx)

            # Waitlists are patients needing matching
            for pid in patients_to_match_ttc:
                current_doctor = matching_dict_ttc[pid]
                # Patients want to match to their highest preference doctor
                highest_pref_doctor = np.argmax(preferences_dict[pid])
                if highest_pref_doctor != current_doctor:
                    if highest_pref_doctor not in waitlists:
                        waitlists[highest_pref_doctor] = []
                    waitlists[highest_pref_doctor].append(pid)

            current_panels, waitlists = run_ttc(current_panels, waitlists)

            # Update matching dictionary
            matching_dict_ttc = {}
            for doctor_idx, patients in current_panels.items():
                for pid in patients:
                    matching_dict_ttc[pid] = doctor_idx

            new_matching_dict_ttc = matching_dict_ttc.copy()

            # find patients whose preferences have been changed
            patients_matched = []
            for pid in patients_to_match_ttc:
                old_doc = old_matching_dict_ttc.get(pid, None)
                new_doc = new_matching_dict_ttc.get(pid, None)
                if new_doc != old_doc:
                    patients_matched.append(pid)

            # remove successfully matched patients
            patient_need_match_ttc -= set(patients_matched)

            end_time_ttc = time.time()
            time_ttc.append(end_time_ttc - start_time_ttc)

            # Compute utility
            utility_t = compute_total_utility(preferences_dict, matching_dict_ttc)
            utility_ttc.append(utility_t)
        else:
            # If no patients need matching, utility remains the same
            utility_ttc.append(utility_ttc[-1])

        # Append current matching to patient history
        for pid in range(num_patients):
            patient_history_ttc[pid].append(matching_dict_ttc.get(pid, None))
        print(f"Number of patients needing TTC match after TTC: {len(patient_need_match_ttc)}")


        # Group 3: No operation
        # Compute utility
        utility_n = compute_total_utility(preferences_dict, matching_dict_no_op)
        utility_no_matching.append(utility_n)

        # Append current matching to patient history
        for pid in range(num_patients):
            patient_history_no_op[pid].append(matching_dict_no_op[pid])
        total_end_time = time.time()
        total_simulation_time.append(total_end_time - total_start_time)


    # Plot utility graph
    plot_utilities(utility_hungarian, utility_ttc, utility_no_matching, num_rounds)
    # Plot execution time graph
    plot_execution_times(time_hungarian, time_ttc, num_rounds,total_simulation_time)

    print("Simulation completed.")
    average_time_hungarian = sum(time_hungarian)/len(time_hungarian)
    print("The average running time for Hungarian Algorithm is:",average_time_hungarian)
    average_time_TTC = sum(time_ttc) / len(time_ttc)
    print("The average running time for TTC Algorithm is:",average_time_TTC)
    average_time_all = sum(total_simulation_time) / len(total_simulation_time)
    print("The average running time for all Algorithm is:",average_time_all)
    return patient_history_hungarian, patient_history_ttc, patient_history_no_op


# Function to track a patient's matching history
def get_patient_matching_history(patient_id, algorithm, patient_history):
    if patient_id not in patient_history:
        raise ValueError(f"Patient ID {patient_id} not found in patient history.")

    history = patient_history[patient_id]
    return history


# Run the simulation and get patient histories
patient_history_hungarian, patient_history_ttc, patient_history_no_op = main()

patient_id = 0    #This is where to change the searching patients.
print(f"\nMatching history for patient {patient_id} using Hungarian Algorithm:")
hungarian_history = get_patient_matching_history(patient_id, 'hungarian', patient_history_hungarian)
print(hungarian_history)

print(f"\nMatching history for patient {patient_id} using TTC Algorithm:")
ttc_history = get_patient_matching_history(patient_id, 'ttc', patient_history_ttc)
print(ttc_history)

print(f"\nMatching history for patient {patient_id} with No Operation:")
no_op_history = get_patient_matching_history(patient_id, 'no_op', patient_history_no_op)
print(no_op_history)
