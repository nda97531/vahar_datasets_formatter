import itertools
import os
import re
import numpy as np
import orjson
import pandas as pd
import polars as pl
from loguru import logger
from glob import glob
from typing import Tuple, Union

from har_datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
from har_datasets.constant import G_TO_MS2, DEG_TO_RAD
from my_py_utils.my_py_utils.pl_dataframe import resample_numeric_df as pl_resample_numeric_df
from my_py_utils.my_py_utils.time_utils import str_2_timestamp


class UPFallConst:
    # the below (m/s^2) and (rad/s) are the processed units, NOT units in the raw data
    INERTIAL_COLS = [
        'timestamp(ms)',

        'ankle_acc_x(m/s^2)', 'ankle_acc_y(m/s^2)', 'ankle_acc_z(m/s^2)',
        'ankle_gyro_x(rad/s)', 'ankle_gyro_y(rad/s)', 'ankle_gyro_z(rad/s)',
        'ankle_illuminance(lx)',

        'pocket_acc_x(m/s^2)', 'pocket_acc_y(m/s^2)', 'pocket_acc_z(m/s^2)',
        'pocket_gyro_x(rad/s)', 'pocket_gyro_y(rad/s)', 'pocket_gyro_z(rad/s)',
        'pocket_illuminance(lx)',

        'belt_acc_x(m/s^2)', 'belt_acc_y(m/s^2)', 'belt_acc_z(m/s^2)',
        'belt_gyro_x(rad/s)', 'belt_gyro_y(rad/s)', 'belt_gyro_z(rad/s)',
        'belt_illuminance(lx)',

        'neck_acc_x(m/s^2)', 'neck_acc_y(m/s^2)', 'neck_acc_z(m/s^2)',
        'neck_gyro_x(rad/s)', 'neck_gyro_y(rad/s)', 'neck_gyro_z(rad/s)',
        'neck_illuminance(lx)',

        'wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)',
        'wrist_gyro_x(rad/s)', 'wrist_gyro_y(rad/s)', 'wrist_gyro_z(rad/s)',
        'wrist_illuminance(lx)',

        'BrainSensor', 'Infrared1', 'Infrared2', 'Infrared3', 'Infrared4', 'Infrared5', 'Infrared6', 'Subject',
        'Activity', 'Trial',

        'label'
    ]
    SELECTED_INERTIAL_COLS = [
        'timestamp(ms)',

        'belt_acc_x(m/s^2)', 'belt_acc_y(m/s^2)', 'belt_acc_z(m/s^2)',
        'belt_gyro_x(rad/s)', 'belt_gyro_y(rad/s)', 'belt_gyro_z(rad/s)',

        'wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)',
        'wrist_gyro_x(rad/s)', 'wrist_gyro_y(rad/s)', 'wrist_gyro_z(rad/s)'
    ]

    SKELETON_COLS = list(itertools.chain.from_iterable(
        [f'x_{joint}', f'y_{joint}'] for joint in
        ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee",
         "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
         "RBigToe", "RSmallToe", "RHeel"]
    ))
    SELECTED_SKELETON_COLS = ['timestamp(ms)'] + list(itertools.chain.from_iterable(
        [f'x_{joint}', f'y_{joint}'] for joint in
        ["Neck", "RElbow", "LElbow", "RWrist", "LWrist", "RKnee", "LKnee", "RAnkle", "LAnkle"]
    ))

    # columns used for skeleton normalisation
    SKELETON_HIP_COLS = ["x_MidHip", "y_MidHip"]

    # modal names
    MODAL_INERTIA = 'inertia'
    MODAL_SKELETON = 'skeleton'

    SKELETON_PANDAS_TIME_FORMAT = '%Y-%m-%dT%H_%M_%S.%f'
    INERTIA_POLARS_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S%.f'
    INERTIAL_POLARS_DTYPES = [pl.Utf8] + [pl.Float64] * 35 + [pl.Int64] * 11

    # minimum confidence threshold to take a skeleton joint
    MIN_JOINT_CONF = 0.05
    # remove every skeleton with its leftmost point not farther to the right than this position
    RIGHT_X = 510
    # remove every skeleton with its lowest point higher than this position
    LOW_Y = 255

    # short activities that can't run sliding window
    SHORT_ACTIVITIES = {1, 2, 3, 4, 5, 9}


class UPFallParquet(ParquetDatasetFormatter):
    """
    Class for processing UP-Fall dataset.
    Use only Inertial sensors and Camera2.
    """

    @staticmethod
    def upper_left_corner_line_equation(x, y):
        """
        Check if joints are in the upper left corner
        """
        return 120 * x + 203 * y - 57449

    @staticmethod
    def read_inertial_and_label(file_path: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Read a sensor csv file, including both inertial data and label

        Args:
            file_path: path to csv file

        Returns:
            2 polars dataframes of the same length: sensor data and label
        """
        data_df = pl.read_csv(file_path, has_header=False, new_columns=UPFallConst.INERTIAL_COLS, skip_rows=2,
                              dtypes=UPFallConst.INERTIAL_POLARS_DTYPES)
        logger.info(f'Raw inertia dataframe: {data_df.shape}')

        # convert timestamp to millisecond integer
        data_df = data_df.with_columns(
            pl.col('timestamp(ms)').str.to_datetime(UPFallConst.INERTIA_POLARS_TIME_FORMAT).dt.timestamp(time_unit='ms')
        )

        # convert units to SI
        data_df = data_df.with_columns([
            pl.col([c for c in UPFallConst.SELECTED_INERTIAL_COLS if c.endswith('(m/s^2)')]) * G_TO_MS2,
            pl.col([c for c in UPFallConst.SELECTED_INERTIAL_COLS if c.endswith('(rad/s)')]) * DEG_TO_RAD
        ])

        label_df = data_df.select(['timestamp(ms)', 'label'])
        data_df = data_df.select(UPFallConst.SELECTED_INERTIAL_COLS)

        return data_df, label_df

    def read_skeleton(self, folder_path: str, norm: bool = True) -> Union[pl.DataFrame, None]:
        """
        Read skeleton data of only the main subject (person) from all frames in a session

        Args:
            folder_path: path to a session folder containing all skeleton json files
            norm: whether to normalise skeleton
                (False only for visualisation, otherwise always True, no need to optimise)

        Returns:
            a polars dataframe, each row is a frame
        """
        json_files = sorted(glob(f'{folder_path}/*.json'))
        logger.info(f'Found {len(json_files)} json files in {folder_path}')

        if not json_files:
            return None
        skel_frames = []
        timestamps = []

        # for each frame in this session
        for json_file in json_files:
            # read a json file
            with open(json_file, 'r') as F:
                skeletons = orjson.loads(F.read())
            skeletons = skeletons['people']
            skeletons = np.array([np.array(skeleton['pose_keypoints_2d']).reshape(-1, 3) for skeleton in skeletons])

            # remove joints with low conf
            low_conf_idx = skeletons[:, :, 2] < UPFallConst.MIN_JOINT_CONF
            skeletons[low_conf_idx, :2] = np.nan

            # select one person as the main subject
            main_skeleton = self.select_main_skeleton(skeletons)

            # remove conf column
            main_skeleton = main_skeleton[:, :2]

            skel_frames.append(main_skeleton.reshape(-1))
            timestamps.append(json_file.split(os.sep)[-1].removesuffix('_keypoints.json'))

        # create DF
        skel_frames = pd.DataFrame(skel_frames, columns=UPFallConst.SKELETON_COLS)
        # add timestamp column
        skel_frames['timestamp(ms)'] = [
            str_2_timestamp(str_time, str_format=UPFallConst.SKELETON_PANDAS_TIME_FORMAT, tz=0)
            for str_time in timestamps
        ]
        # only keep representative joints, but also include 2 joints for normalisation
        skel_frames = skel_frames[UPFallConst.SELECTED_SKELETON_COLS + UPFallConst.SKELETON_HIP_COLS]

        # fill low conf joints by interpolation
        skel_frames = skel_frames.interpolate()
        skel_frames = skel_frames.fillna(method='backfill')

        # mid hip is always at origin (0, 0)
        assert not pd.isna(skel_frames.iloc[0, -len(UPFallConst.SKELETON_HIP_COLS):]).any(), 'Norm joints are nan!!'
        norm_midhip_point = skel_frames.iloc[:, -len(UPFallConst.SKELETON_HIP_COLS):].to_numpy()

        skel_frames = pl.from_pandas(skel_frames.iloc[:, :-len(UPFallConst.SKELETON_HIP_COLS)])
        if norm:
            # skeleton size is always 1
            x_cols = [c for c in skel_frames.columns if c.startswith('x_')]
            y_cols = [c for c in skel_frames.columns if c.startswith('y_')]
            x_arr = skel_frames.select(x_cols).to_numpy()
            y_arr = skel_frames.select(y_cols).to_numpy()
            upper_y = np.nanpercentile(y_arr, q=75, axis=1)
            lower_y = np.nanpercentile(y_arr, q=25, axis=1)
            upper_x = np.nanpercentile(x_arr, q=75, axis=1)
            lower_x = np.nanpercentile(x_arr, q=25, axis=1)
            skeleton_size = np.sqrt((upper_x - lower_x) ** 2 + (upper_y - lower_y) ** 2) + 1e-3

            # normalise skeleton session
            skel_frames = skel_frames.with_columns(
                (pl.col(x_cols) - norm_midhip_point[:, 0]) / skeleton_size,
                (pl.col(y_cols) - norm_midhip_point[:, 1]) / skeleton_size
            )
        # fill undetected joints
        skel_frames = skel_frames.with_columns(
            pl.exclude('timestamp(ms)').fill_null(0.)
        )
        return skel_frames

    def skeletons_in_upper_left_corner(self, skeletons: np.ndarray) -> np.ndarray:
        """
        Check if skeletons are in the upper left corner of the frame

        Args:
            skeletons: 3d array shape [num skeleton, num joint, 3(x, y, conf)]

        Returns:
            1d array shape [num skeleton] containing boolean values, True if a skeleton is in the corner
        """
        relative_positions = self.upper_left_corner_line_equation(skeletons[:, :, 0], skeletons[:, :, 1])
        skeleton_in_corner = (relative_positions <= 0) | np.isnan(relative_positions)
        skeleton_in_corner = skeleton_in_corner.all(axis=1)
        return skeleton_in_corner

    def select_main_skeleton(self, skeletons: np.ndarray) -> np.ndarray:
        """
        Select 1 skeleton as the main subject from a list of skeletons in 1 frame.
        All conditions are specialised for Camera2, UP-Fall dataset.

        Args:
            skeletons: 3d array shape [num skeletons, num joints, 3(x, y, confidence score)]

        Returns:
            the selected skeleton, 2d array shape [num joints, 3(x, y, confidence score)]
        """
        if len(skeletons) == 0:
            raise ValueError('empty skeleton list')

        # return if there's only 1 skeleton in the list
        if len(skeletons) == 1:
            return skeletons[0]

        # Condition 1: exclude people on the right half of the frame
        # find the leftmost joint of every skeleton
        skel_xs = np.nanmin(skeletons[:, :, 0], axis=1)
        # remove skeletons that are too far to the right
        skeletons = skeletons[skel_xs <= UPFallConst.RIGHT_X]

        # Condition 2: exclude people on the upper half of the frame (too far from the camera)
        # find the lowest joint of every skeleton
        skel_ys = np.nanmax(skeletons[:, :, 1], axis=1)
        # remove skeletons that are too high (too far from the camera)
        skeletons = skeletons[skel_ys >= UPFallConst.LOW_Y]

        # Condition 3: exclude people in the upper left corner (the corridor)
        # check if joints are in the corner
        skeletons_in_corner = self.skeletons_in_upper_left_corner(skeletons)
        # remove skeletons with all joints in the corner
        skeletons = skeletons[~skeletons_in_corner]

        # the rest: return max conf
        if len(skeletons) == 1:
            return skeletons[0]
        if len(skeletons) == 0:
            return np.concatenate([np.full([25, 2], fill_value=np.nan), np.zeros([25, 1])], axis=-1)

        mean_conf = skeletons[:, :, 2].mean()
        most_conf_skeleton = skeletons[np.argmax(mean_conf)]
        return most_conf_skeleton

    @staticmethod
    def get_info_from_session_folder(path: str) -> tuple:
        """
        Get info from raw session folder path using regex

        Args:
            path: session folder path

        Returns:
            a tuple of integers: [subject, activity, trial]
        """
        info = re.search(r'Subject([0-9]*)/Activity([0-9]*)/Trial([0-9]*)', path)
        info = tuple(int(info.group(i)) for i in range(1, 4))
        return info

    def process_session(self, session_folder: str, session_info: str = None) \
            -> Union[Tuple[pl.DataFrame, pl.DataFrame], None]:
        """
        Fully process dataframe for each modal in a session

        Args:
            session_folder: path to session folder
            session_info: session ID

        Returns:
            2 DFs: inertia, skeleton
        """
        if not session_info:
            subject, activity, trial = self.get_info_from_session_folder(session_folder)
            session_info = f'Subject{subject}Activity{activity}Trial{trial}'

        # read data from files
        skeleton_df = self.read_skeleton(f'{session_folder}/{session_info}Camera2/skeleton/')
        if skeleton_df is None:
            return None
        inertial_df, label_df = self.read_inertial_and_label(f'{session_folder}/{session_info}.csv')

        # re-sample
        inertial_df = pl_resample_numeric_df(inertial_df, 'timestamp(ms)',
                                             self.sampling_rates[UPFallConst.MODAL_INERTIA])
        skeleton_df = pl_resample_numeric_df(skeleton_df, 'timestamp(ms)',
                                             self.sampling_rates[UPFallConst.MODAL_SKELETON])

        # assign labels
        inertial_df = inertial_df.set_sorted('timestamp(ms)')
        skeleton_df = skeleton_df.set_sorted('timestamp(ms)')
        label_df = label_df.set_sorted('timestamp(ms)')
        inertial_df = inertial_df.join_asof(label_df, on='timestamp(ms)', strategy='nearest')
        skeleton_df = skeleton_df.join_asof(label_df, on='timestamp(ms)', strategy='nearest')

        logger.info(f'Processed inertial dataframe: {inertial_df.shape};\t'
                    f'Processed skeletal dataframe: {skeleton_df.shape}')

        return inertial_df, skeleton_df

    def run(self):
        logger.info('Scanning for session folders...')
        session_folders = sorted(glob(f'{self.raw_folder}/Subject*/Activity*/Trial*'))
        logger.info(f'Found {len(session_folders)} sessions in total')

        skip_file = 0
        write_file = 0
        # for each session
        for session_folder in session_folders:
            # get session info
            subject, activity, trial = self.get_info_from_session_folder(session_folder)
            session_info = f'Subject{subject}Activity{activity}Trial{trial}'

            if os.path.isfile(self.get_output_file_path(UPFallConst.MODAL_INERTIA, subject, session_info)) \
                    and os.path.isfile(self.get_output_file_path(UPFallConst.MODAL_SKELETON, subject, session_info)):
                logger.info(f'Skipping session {session_info} because already run before')
                skip_file += 2
                continue
            logger.info(f'Starting session {session_info}')

            # get data
            data = self.process_session(session_folder, session_info)
            if data is None:
                logger.info(f'Skipping session {session_info} because skeleton data not found')
                skip_file += 2
                continue
            inertial_df, skeleton_df = data

            # write files
            if self.write_output_parquet(inertial_df, UPFallConst.MODAL_INERTIA, subject, session_info):
                write_file += 1
            else:
                skip_file += 1
            if self.write_output_parquet(skeleton_df, UPFallConst.MODAL_SKELETON, subject, session_info):
                write_file += 1
            else:
                skip_file += 1
        logger.info(f'{write_file} file(s) written, {skip_file} file(s) skipped')


class UPFallNpyWindow(NpyWindowFormatter):
    def run(self, shift_short_activity: bool = True) -> pd.DataFrame:
        """

        Args:
            shift_short_activity: whether to run shifting windows on fall sessions (or just run normal sliding window)

        Returns:
            please see parent class's method
        """
        # get list of parquet files
        parquet_sessions = self.get_parquet_file_list()

        result = []
        # for each session
        for parquet_session in parquet_sessions.iter_rows(named=True):
            # get session info
            _, subject, session_id = self.get_parquet_session_info(list(parquet_session.values())[0])
            session_regex = re.match('Subject(?:[0-9]*)Activity([0-9]*)Trial([0-9]*)', session_id)
            session_label, session_trial = (int(session_regex.group(i)) for i in [1, 2])

            session_result = self.process_parquet_to_windows(
                parquet_session=parquet_session,
                subject=subject,
                session_label=session_label,
                is_short_activity=(session_label in UPFallConst.SHORT_ACTIVITIES) if shift_short_activity else False
            )

            result.append(session_result)
        result = pd.DataFrame(result)
        return result


if __name__ == '__main__':
    parquet_dir = '/mnt/data_drive/projects/UCD04 - Virtual sensor fusion/processed_parquet/UP-Fall_draft'
    inertial_freq = 50
    skeletal_freq = 20
    window_size_sec = 4
    step_size_sec = 1.5
    min_step_size_sec = 0.5

    UPFallParquet(
        raw_folder='/mnt/data_drive/projects/raw datasets/UP-Fall',
        destination_folder=parquet_dir,
        sampling_rates={UPFallConst.MODAL_INERTIA: inertial_freq,
                        UPFallConst.MODAL_SKELETON: skeletal_freq}
    ).run()

    # upfall = UPFallNpyWindow(
    #     parquet_root_dir=parquet_dir,
    #     window_size_sec=window_size_sec,
    #     step_size_sec=step_size_sec,
    #     min_step_size_sec=min_step_size_sec,
    #     max_short_window=3,
    #     modal_cols={
    #         UPFallConst.MODAL_INERTIA: {
    #             'wrist_acc': ['wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)'],
    #             'belt_acc': ['belt_acc_x(m/s^2)', 'belt_acc_y(m/s^2)', 'belt_acc_z(m/s^2)']
    #         },
    #         UPFallConst.MODAL_SKELETON: {
    #             'skeleton': None
    #         }
    #     }
    # ).run()
