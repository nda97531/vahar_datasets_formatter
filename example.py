from vahar.datasets.sfu_dataset import SFUParquet, SFUConst, SFUNpyWindow

smu_parquet_folder = 'parquet_dataset/SFU-IMU'

# process raw dataset to the intended parquet format
SFUParquet(
    raw_folder='raw_dataset/SFU-IMU',
    destination_folder=smu_parquet_folder,
    sampling_rates={SFUConst.MODAL: 50}
).run()

# sliding window
windows = SFUNpyWindow(
    parquet_root_dir=smu_parquet_folder,
    window_size_sec=4, step_size_sec=2, min_step_size_sec=0.5, max_short_window=5,
    modal_cols={
        'inertia': {
            pos: [F'{pos}_acc_{axis}(m/s^2)' for axis in ['x', 'y', 'z']]
            for pos in SFUConst.SENSOR_POSITIONS
        }
    }
).run()
