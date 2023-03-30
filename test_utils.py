import time
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Pool
from multiprocessing import cpu_count
import random


def run_sims(iterations):
    sim_list = []
    for i in range(iterations):
        sim_list.append(random.uniform(0, 1))
    print(iterations, "count", sum(sim_list)/len(sim_list))
    return (sum(sim_list)/len(sim_list))


def worker(queue):
    i = 0
    while not queue.empty():
        task = queue.get()
        run_sims(task)
        i = i+1


if __name__ == '__main__':

    iteration_count = 5

    queue = Queue()
    iterations_list = [30000000] * iteration_count
    it_len = len(iterations_list)

    # guess a parallel execution size. CPU bound, and we want some
    # room for other processes.
    pool_size = max(min(cpu_count()-2, len(iterations_list)), 2)
    print("Pool size", pool_size)

    ## Queue ##
    print("#STARTING QUEUE#")
    start_t = time.perf_counter()
    for iterations in iterations_list:
        queue.put(iterations)

    processes = []
    for i in range(pool_size):
        processes.append(Process(target=worker, args=(queue, )))
        processes[-1].start()
    for process in processes:
        process.join()
    end_t = time.perf_counter()
    print("Queue time: ", end_t - start_t)

    ## Pool ##
    print("#STARTING POOL#")
    start_t = time.perf_counter()
    with Pool(pool_size) as pool:
        results = pool.imap_unordered(run_sims, iterations_list)

        for res in results:
            res
    end_t = time.perf_counter()
    print("Pool time: ", end_t - start_t)

    # No Multiprocessing - Normal Loop
    print("#STARTING NORMAL LOOP#")
    start_t = time.perf_counter()
    for i in iterations_list:
        run_sims(i)
    end_t = time.perf_counter()
    print("Normal time: ", end_t - start_t)
