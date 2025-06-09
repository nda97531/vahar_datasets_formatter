import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import polars as pl
import numpy as np

ske_file = '/home/nda97531/Documents/datasets/dataset_parquet/cmdfull/skeleton/subject_7/S30P7.parquet'
ine_file = ske_file.replace('/skeleton/', '/inertia/')

ske_df = pl.read_parquet(ske_file)
ine_df = pl.read_parquet(ine_file)

assert ske_df.columns[0] == 'timestamp(ms)' and ske_df.columns[-1] == 'label'
assert ine_df.columns[0] == 'timestamp(ms)' and ine_df.columns[-1] == 'label'

ske_arr = ske_df.to_numpy()
ine_arr = ine_df.to_numpy()

# convert to the same frequency
ine_arr = ine_arr[np.linspace(start=0, stop=len(ine_arr) - 1, num=len(ske_arr), endpoint=True, dtype=int)]

# split label and data into separate arrays
lbl_arr = ine_arr[:, [0, -1]]
ine_arr = ine_arr[:, 1:-1]

# reshape skeleton array
ske_arr = ske_arr[:, 1:-1].reshape(len(ske_arr), 7, 20, 3)  # 7 kinect, 20 joints, 3 axes

# start where kinect 7 has data
first_ske_idx = (
    ~(np.isnan(ske_arr[:, -1:]).all(axis=(1, 2, 3)))
).nonzero()[0][0]
lbl_arr = lbl_arr[first_ske_idx:]
ine_arr = ine_arr[first_ske_idx:]
ske_arr = ske_arr[first_ske_idx:]

# create window margin for inertial data
half_window_len = 20
ine_arr_expand = np.zeros([len(ine_arr) + half_window_len * 2, 6])
ine_arr_expand[half_window_len:-half_window_len] = ine_arr[:, :6]

# create a figure with 9 subplots
# fig, axs = plt.subplots(3, 3)
fig = plt.figure()
axs = []
for i in range(2):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.set_title(['waist', 'wrist'][i])
    axs.append(ax)
for i in range(2, 9):
    ax = fig.add_subplot(3, 3, i + 1, projection='3d')
    ax.set_title(f'kinect{i - 1}')
    axs.append(ax)  # enable 3D plot

# initialise objects (one in each axis); 2 inertial sensors, 7 kinect
plot_acc = [
    axs[i // 3].plot([], [], lw=2)[0]
    for i in range(2 * 3)
]
plot_ske = [
    axs[i].scatter([], [], [])
    for i in range(2, 9)
]
plots = plot_acc + plot_ske

label = axs[0].text(0.2, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                    transform=axs[0].transAxes, ha="center")

for i in range(2):
    axs[i].set_ylim(-20, 20)
    axs[i].set_xlim(0, 41)
    axs[i].grid()

for i in range(2, 9):
    axs[i].set_ylim(-1.5, 1.5)
    axs[i].set_xlim(-1.5, 1.5)
    axs[i].set_zlim(-0.1, 2.0)
    axs[i].grid()


def update(frame_idx):
    # update the accelerometer window
    window_data = ine_arr_expand[frame_idx:frame_idx + half_window_len * 2]
    for i in range(2 * 3):
        plots[i].set_data(np.arange(len(window_data)), window_data[:, i])

    # update skeleton scatter
    joints = ske_arr[frame_idx]  # shape (kinect, joint, 3)
    for i in range(7):
        plot_idx = len(plot_acc) + i
        plots[plot_idx]._offsets3d = (joints[i, :, 0], joints[i, :, 1] * -1, joints[i, :, 2])

    label.set_text(f'label: {int(lbl_arr[frame_idx, 1])}')
    return plots


ani = FuncAnimation(fig, update, interval=1, frames=len(ske_arr), repeat=False)
plt.show()
