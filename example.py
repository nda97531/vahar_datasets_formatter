from har_datasets.sfu_dataset import SFUParquet, SFUConst, SFUNpyWindow

SFUParquet(
    raw_folder='raw_dataset/SFU-IMU',
    destination_folder='parquet_dataset/SFU-IMU',
    sampling_rates={SFUConst.MODAL: 50}
).run()

SFUNpyWindow(
    parquet_root_dir='/mnt/data_partition/UCD/UCD04 - Virtual sensor fusion/processed_parquet/SFU-IMU',
    window_size_sec=4, step_size_sec=2, min_step_size_sec=0.5, max_short_window=5,
    modal_cols={
        'inertia': {
            pos: [F'{pos}_acc_{axis}(m/s^2)' for axis in ['x', 'y', 'z']]
            for pos in SFUConst.SENSOR_POSITIONS
        }
    }
).run()
