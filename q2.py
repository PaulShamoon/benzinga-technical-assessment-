import heapq
from collections import defaultdict

Task = tuple[int, int, int, list[int]]
Assignment = tuple[int, int, int]


def _build_graph(
    tasks: list[Task]
) -> tuple[dict[int, int], defaultdict[int, list[int]]]:
    """Helper method to build the dependency bookkeeping for the DAG

    Args:
        tasks (list[Task]): (id, duration, priority, deps) tuples

    Returns:
        tuple[dict[int, int], defaultdict[int, list[int]]]: Number of unfinished
            deps per task id, as well as a map from each task id to the ids that depend on itt
    """
    remaining_deps = {task_id: len(deps) for task_id, _, _, deps in tasks}
    dependents = defaultdict(list)

    for task_id, _, _, deps in tasks:
        for dep in deps:
            dependents[dep].append(task_id)

    return remaining_deps, dependents


def _assign_ready_tasks(
    time: int,
    ready: list[tuple[int, int]],
    free_workers: list[int],
    running: list[tuple[int, int, int]],
    durations: dict[int, int],
    schedule: list[Assignment]
) -> None:
    """Helper method to start as many ready tasks as there are free workers, highest priority first

    Args:
        time (int): The current event time
        ready (list[tuple[int, int]]): Min heap of (-priority, task_id)
        free_workers (list[int]): Min heap of free worker indexes
        running (list[tuple[int, int, int]]): Min heap of (end_time, worker, task_id)
        durations (dict[int, int]): Duration of each task id
        schedule (list[Assignment]): Started tasks as (task_id, start_time, worker)
    """
    while ready and free_workers:
        _, task_id = heapq.heappop(ready)
        worker = heapq.heappop(free_workers)

        schedule.append((task_id, time, worker))
        heapq.heappush(running, (time + durations[task_id], worker, task_id))


def _complete_tasks(
    time: int,
    running: list[tuple[int, int, int]],
    free_workers: list[int],
    remaining_deps: dict[int, int],
    dependents: defaultdict[int, list[int]],
    priorities: dict[int, int],
    ready: list[tuple[int, int]]
) -> None:
    """Helper method to finish every task ending at 'time' and unlock dependents

    Args:
        time (int): The current event time
        running (list[tuple[int, int, int]]): Min heap of (end_time, worker, task_id)
        free_workers (list[int]): Min heap of free worker indices, gains the workers that just finished
        remaining_deps (dict[int, int]): Unfinished dep count per task id
        dependents (defaultdict[int, list[int]]): For each task id, the ids that depend on it
        priorities (dict[int, int]): Priority of each task id
        ready (list[tuple[int, int]]): Min heap of (-priority, task_id), gains the dependents that just became ready
    """
    while running and running[0][0] == time:
        _, worker, task_id = heapq.heappop(running)
        heapq.heappush(free_workers, worker)

        for dependent in dependents[task_id]:
            remaining_deps[dependent] -= 1
            if remaining_deps[dependent] == 0:
                heapq.heappush(ready, (-priorities[dependent], dependent))


def schedule_tasks(
    num_workers: int,
    tasks: list[Task]
) -> tuple[int, list[Assignment]]:
    """Simulates greedy execution of a task DAG on a pool of identical workers

    Args:
        num_workers (int): Number of parallel worker agents (W)
        tasks (list[Task]): (id, duration, priority, deps) tuples

    Returns:
        tuple[int, list[Assignment]]: The makespan, and the schedule as
            (task_id, start_time, worker_index) sorted by start time then task id
    """
    durations = {task_id: duration for task_id, duration, _, _ in tasks}
    priorities = {task_id: priority for task_id, _, priority, _ in tasks}
    remaining_deps, dependents = _build_graph(tasks)

    ready = [
        (-priorities[task_id], task_id)
        for task_id, count in remaining_deps.items()
        if count == 0
    ]
    heapq.heapify(ready)

    free_workers = list(range(num_workers))
    running: list[tuple[int, int, int]] = []
    schedule: list[Assignment] = []
    time = 0

    while True:
        _assign_ready_tasks(
            time,
            ready,
            free_workers,
            running,
            durations,
            schedule
        )

        if not running:
            break

        time = running[0][0]
        _complete_tasks(
            time,
            running,
            free_workers,
            remaining_deps,
            dependents,
            priorities,
            ready
        )

    schedule.sort(key=lambda assignment: (assignment[1], assignment[0]))
    return time, schedule


if __name__ == "__main__":
    example_tasks = [
        (1, 3, 5, []),
        (2, 2, 1, []),
        (3, 4, 9, [1]),
        (4, 1, 1, [1, 2]),
    ]
    example_makespan, example_schedule = schedule_tasks(2, example_tasks)
    print(example_makespan)
    print(example_schedule)

    # Other test cases

    # example_tasks = [
    #     (1, 2, 1, []),
    #     (2, 3, 5, []),
    #     (3, 1, 5, []),
    # ]

    # example_tasks = [
    #     (1, 4, 1, []),
    #     (2, 2, 1, []),
    # ]

    # example_tasks = []
