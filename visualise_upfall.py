import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

ske_file = '/mnt/data_partition/UCD/dataset_processed/UP-Fall_new/skeleton/subject_12/Subject12Activity1Trial1.parquet'
ine_file = ske_file.replace('/skeleton/', '/inertia/')

ske_df = pd.read_parquet(ske_file)
ine_df = pd.read_parquet(ine_file)

ske_arr = ske_df.to_numpy()
ine_arr = ine_df.to_numpy()

# convert to the same frequency
ine_arr = ine_arr[np.linspace(start=0, stop=len(ine_arr) - 1, num=len(ske_arr), endpoint=True, dtype=int)]
# split label and data into separate arrays
lbl_arr = ine_arr[:, [0, -1]]
ine_arr = ine_arr[:, 1:-1]
# reshape skeleton array
ske_arr = ske_arr[:, 1:-1].reshape(-1, 9, 2)
# flip Y axis
ske_arr[:, :, 1] *= -1
# create window margin for inertial data
half_window_len = 20
ine_arr_expand = np.zeros([len(ine_arr) + half_window_len * 2, 3])
ine_arr_expand[half_window_len:-half_window_len] = ine_arr[:, :3]

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

# initialise objects (one in each axis)
plot_acc = [ax1.plot([], [], lw=2)[0] for _ in range(3)]
plot_ske = ax2.scatter([], [])
plots = [*plot_acc, plot_ske]

label = ax1.text(0.1, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                 transform=ax1.transAxes, ha="center")

ax1.set_ylim(-15, 15)
ax1.set_xlim(0, 41)
ax1.grid()

ax2.set_ylim(-3, 3)
ax2.set_xlim(-3, 3)
ax2.grid()


def update(frame_idx):
    # update the data of both objects
    window_data = ine_arr_expand[frame_idx:frame_idx + half_window_len * 2]
    for axis in range(3):
        plots[axis].set_data(np.arange(len(window_data)), window_data[:, axis])
    plots[3].set_offsets(ske_arr[frame_idx])
    label.set_text(str(lbl_arr[frame_idx, 1]))
    return plots


ani = FuncAnimation(fig, update,  interval=50, frames=len(ske_arr), repeat=False)
plt.show()
