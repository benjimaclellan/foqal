import itertools

from foqal.utils.io import IO
from foqal.utils.constants import states, channels


if __name__ == "__main__":

    io = IO.directory(
        folder="gpt-generated-data", verbose=False, include_date=False, include_uuid=False
    )
    for run in (0, 1, 2):
        num = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

        for i, (num_states, num_effects) in enumerate(zip(num, num)):
            print(f"Run {run} | {i} of {len(num)} | m = {m}, p = {p}")
            state = channels['depolarized'](p=p)
            nd_array = simulate_dataset(m=m, state=state)

            io.save_np_array(nd_array.astype("float"), filename=f"m={m}_p={p}_run{run}")
