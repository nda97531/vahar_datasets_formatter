import os
import re
from collections import defaultdict
from copy import deepcopy
from glob import glob
from typing import List, Dict
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from transforms3d.axangles import axangle2mat

from my_py_utils.my_py_utils.string_utils import rreplace

if __name__ == '__main__':
    from vahar.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from vahar.constant import G_TO_MS2
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from ..constant import G_TO_MS2


def merge_annotation(anno_dfs: list[pl.DataFrame], start_col='start', end_col='end', label_col='label') -> pl.DataFrame:
    """
    Merge multiple camera annotation dataframes into a single dataframe.

    Args:
        anno_dfs: list of polars DataFrames. Each DataFrame has columns [label_col, start_col, end_col].
        start_col: start timestamp column name
        end_col: end timestamp column name
        label_col: label column name

    Returns:
        A merged DataFrame with columns [start_col, end_col, label_col].
    """
    if len(anno_dfs) == 1:
        return anno_dfs[0].select(start_col, end_col, label_col)

    # get all unique time boundaries
    boundaries = set()
    for df in anno_dfs:
        boundaries.update(df.get_column(start_col))
        boundaries.update(df.get_column(end_col))
    time_points = sorted(boundaries)

    # generate non-overlapping intervals
    merged_rows = []
    for i in range(len(time_points) - 1):
        interval_start = time_points[i]
        interval_end = time_points[i + 1]
        interval_labels = []

        for cam_idx, df in enumerate(anno_dfs):
            # check if this interval overlaps with any annotation in this DF
            overlapping = df.filter((pl.col(start_col) <= interval_start) & (interval_end <= pl.col(end_col)))
            if len(overlapping) > 0:
                interval_labels.append(overlapping.item(0, label_col))

        # apply label merging logic
        if len(interval_labels) == 0:
            merged_label = None
        elif len(set(interval_labels)) == 1:
            merged_label = interval_labels[0]
        else:
            print(
                f"Warning: conflicting labels {interval_labels} between {interval_start} and {interval_end}. Marked as inactivity.")
            merged_label = None

        if merged_label is not None:
            if (len(merged_rows) > 0) and (merged_label == merged_rows[-1][label_col]) and (
                    interval_start == merged_rows[-1][end_col]):
                merged_rows[-1][end_col] = interval_end
            else:
                merged_rows.append({start_col: interval_start, end_col: interval_end, label_col: merged_label})

    return pl.DataFrame(merged_rows)


class CMDFallConst:
    # modal names
    MODAL_INERTIA = 'inertia'
    MODAL_SKELETON = 'skeleton'

    # raw sampling rates in Hertz
    RAW_INERTIA_FREQ = 50
    RAW_KINECT_FPS = 20

    ACCELEROMETER_POSITION = {1: 'wrist', 155: 'waist'}

    LABEL_DICT = {0: 'unknown', 1: 'walk', 2: 'run_slowly', 3: 'static_jump', 4: 'move_hand_and_leg',
                  5: 'left_hand_pick_up', 6: 'right_hand_pick_up', 7: 'stagger', 8: 'front_fall', 9: 'back_fall',
                  10: 'left_fall', 11: 'right_fall', 12: 'crawl', 13: 'sit_on_chair_then_stand_up', 14: 'move_chair',
                  15: 'sit_on_chair_then_fall_left', 16: 'sit_on_chair_then_fall_right', 17: 'sit_on_bed_and_stand_up',
                  18: 'lie_on_bed_and_sit_up', 19: 'lie_on_bed_and_fall_left', 20: 'lie_on_bed_and_fall_right'}

    JOINTS_LIST = [
        'hipCenter', 'spine', 'shoulderCenter', 'head',
        'leftShoulder', 'leftElbow', 'leftWrist', 'leftHand',
        'rightShoulder', 'rightElbow', 'rightWrist', 'rightHand',
        'leftHip', 'leftKnee', 'leftAnkle', 'leftFoot',
        'rightHip', 'rightKnee', 'rightAnkle', 'rightFoot'
    ]
    SELECTED_JOINT_LIST = JOINTS_LIST.copy()

    SELECTED_JOINT_IDX: List[int]
    SELECTED_SKELETON_COLS: List[str]
    SKELETON_ROT_MAT: Dict[int, np.ndarray]

    @classmethod
    def define_att(cls):
        cls.SELECTED_JOINT_IDX = [cls.JOINTS_LIST.index(item) for item in cls.SELECTED_JOINT_LIST]
        cls.SELECTED_SKELETON_COLS = [f'kinect{{kinect_id}}_{joint}_{axis}'
                                      for joint in cls.SELECTED_JOINT_LIST
                                      for axis in ['x', 'y', 'z']]

        # floor equation, mean of all non-zero equations in raw data
        # key: kinect ID, value: equation coefficients [a, b, c, d]
        floor_eqs = {
            1: [0.08206175, 0.93994347, -0.33037784, 1.70011137],
            2: [0.01397648, 0.91808462, -0.39580822, 1.67926973],
            3: [0.0262718, 0.90158846, -0.43152657, 1.63472081],
            4: [-0.01118109, 0.92673825, -0.37469574, 1.71229848],
            5: [-0.04063133, 0.97007644, -0.23862125, 1.80490598]
        }
        floor_eqs = {kinect_id: np.array(eq)[[0, 2, 1, 3]]
                     for kinect_id, eq in floor_eqs.items()}
        cls.SKELETON_ROT_MAT = {}
        for kinect_id, floor_eq in floor_eqs.items():
            norm2 = np.linalg.norm(floor_eq[:3])
            rot_angle = np.arccos(floor_eq[2] / norm2)
            rot_axis = np.cross(floor_eq[:3], [0, 0, 1]) / norm2
            cls.SKELETON_ROT_MAT[kinect_id] = axangle2mat(rot_axis, rot_angle)


CMDFallConst.define_att()


class CMDFallParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 min_length_segment: float = 10):
        """
        Class for processing CMDFall dataset.
        Use only Inertial sensors and Camera 3.
        Labels:
        - 0: unknown activity
        - from 1 to 20: as in the raw dataset

        Args:
            min_length_segment: only write segments longer than this threshold (unit: sec)
            use_accelerometer: inertial sensor IDs
        """
        super().__init__(raw_folder, destination_folder, sampling_rates)

        self.min_length_segment = min_length_segment * 1000
        self.use_accelerometer = (1, 155)
        self.use_kinect = (1, 2, 3, 4, 5, 6, 7)

        # if actual interval > expected interval * this coeff; it's considered an interruption and DF will be split
        max_interval_coeff = 4
        # maximum intervals in millisecond
        self.max_interval = {
            CMDFallConst.MODAL_INERTIA: 1000 / CMDFallConst.RAW_INERTIA_FREQ * max_interval_coeff,
            CMDFallConst.MODAL_SKELETON: 1000 / CMDFallConst.RAW_KINECT_FPS * max_interval_coeff
        }

        # read annotation file
        self.anno_df = pl.read_csv(f'{raw_folder}/annotation.csv')

    def scan_data_files(self) -> pd.DataFrame:
        """
        Scan all data files

        Returns:
            a DF, each row is a session, each column is a sensor; column name format: '{sensor type}_{sensor id}'
        """
        # scan files of the first accelerometer
        first_inertia_name = f'{CMDFallConst.MODAL_INERTIA}_{self.use_accelerometer[0]}'
        session_files = {
            first_inertia_name: sorted(
                glob(f'{self.raw_folder}/accelerometer/*I{self.use_accelerometer[0]}.txt'))
        }
        # generate file names for other accelerometer(s)
        old_path_tail = f'I{self.use_accelerometer[0]}.txt'
        for inertia_sensor_id in self.use_accelerometer[1:]:
            new_path_tail = f'I{inertia_sensor_id}.txt'
            session_files[f'{CMDFallConst.MODAL_INERTIA}_{inertia_sensor_id}'] = [
                rreplace(path, old_path_tail, new_path_tail)
                for path in session_files[first_inertia_name]
            ]

        # generate file names for skeleton files
        for kinect_id in self.use_kinect:
            new_path_tail = f'K{kinect_id}.txt'
            session_files[f'{CMDFallConst.MODAL_SKELETON}_{kinect_id}'] = [
                rreplace(rreplace(path, old_path_tail, new_path_tail), 'accelerometer', 'skeleton')
                for path in session_files[first_inertia_name]
            ]

        session_files = pd.DataFrame(session_files)
        return session_files

    @staticmethod
    def get_info_from_session_file(path: str) -> tuple:
        """
        Get info from data file path

        Args:
            path: path to raw accelerometer or skeleton file

        Returns:
            (session ID, subject ID, sensor ID)
        """
        info = re.search(r'S([0-9]*)P([0-9]*)[IK]([0-9]*).txt', path.split('/')[-1])
        info = tuple(int(info.group(i)) for i in range(1, 4))
        return info

    @staticmethod
    def read_accelerometer_df_file(path: str) -> pl.DataFrame:
        """
        Read and format accelerometer file (convert unit, change column names)

        Args:
            path: path to file

        Returns:
            a polars dataframe
        """
        session_id, subject, sensor_id = CMDFallParquet.get_info_from_session_file(path)
        sensor_pos = CMDFallConst.ACCELEROMETER_POSITION[sensor_id]
        df = pl.read_csv(path, columns=['timestamp', 'x', 'y', 'z'])

        df = df.with_columns(pl.col(['x', 'y', 'z']) * G_TO_MS2)

        df = df.rename({
            'timestamp': 'timestamp(ms)',
            'x': f'{sensor_pos}_acc_x(m/s^2)', 'y': f'{sensor_pos}_acc_y(m/s^2)', 'z': f'{sensor_pos}_acc_z(m/s^2)',
        })

        return df

    @staticmethod
    def normalise_skeletons(skeletons: np.ndarray, kinect_id: int) -> np.ndarray:
        """
        Normalise skeletons

        Args:
            skeletons: 3D array shape [frame, joint, axis]
            kinect_id: kinect ID

        Returns:
            array of the same shape
        """
        # remove unused joints
        skeletons = skeletons[:, CMDFallConst.SELECTED_JOINT_IDX, :]

        # straighten skeleton (rotate so that it stands up right)
        skeletons = skeletons.transpose([0, 2, 1])
        if kinect_id in CMDFallConst.SKELETON_ROT_MAT:
            skeletons = np.matmul(CMDFallConst.SKELETON_ROT_MAT[kinect_id], skeletons)
        skeletons = skeletons.transpose([0, 2, 1])

        # move skeleton to coordinate origin
        centre_xy = skeletons[:, :, :2].mean(axis=1, keepdims=True)
        lowest_z = np.percentile(skeletons[:, :, 2], q=10)

        skeletons[:, :, :2] -= centre_xy
        skeletons[:, :, -1] -= lowest_z

        return skeletons

    @staticmethod
    def read_skeleton_df_file(path: str) -> pl.DataFrame:
        """
        Read and format skeleton file (normalise, change column names)

        Args:
            path: path to file

        Returns:
            a polars dataframe
        """
        session_id, subject, sensor_id = CMDFallParquet.get_info_from_session_file(path)

        # raw columns: timestamp,frame_index,person_index,skeleton_data[20jointsx5(X,Y,Z,Xrgb,Yrgb)],floor_equation
        df = pl.read_csv(path, skip_rows=1, has_header=False)
        data_df = df.get_column('column_4')
        info_df = df.select(pl.col('column_1').alias('timestamp(ms)'), pl.col('column_2').alias('frame_index'))
        del df

        # shape [frame, joint * axis]
        data_df = np.array([s.strip().split(' ') for s in data_df], dtype=float)
        org_length = len(data_df)

        # remove 2 RGB columns, keep 3D columns
        # shape [frame, joint, axis]
        data_df = data_df.reshape([org_length, len(CMDFallConst.JOINTS_LIST), 5])
        data_df = data_df[:, :, :3]
        # switch Y and Z
        data_df = data_df[:, :, [0, 2, 1]]
        # normalise skeleton
        data_df = CMDFallParquet.normalise_skeletons(data_df, sensor_id)

        # shape [frame, joint * axis]
        data_df = data_df.reshape([org_length, len(CMDFallConst.SELECTED_JOINT_LIST) * 3])
        data_df = pl.DataFrame(data_df,
                               schema=[c.format(kinect_id=sensor_id) for c in CMDFallConst.SELECTED_SKELETON_COLS])

        df = pl.concat([info_df, data_df], how='horizontal')
        return df

    def resample_session_dfs(self, data_dfs: dict) -> dict[str, pl.DataFrame]:
        """
        Resample session dataframes of all views so they match in duration and timestamps.

        Args:
            data_dfs: a dict with keys are sensor (format: '{sensor type}_{sensor id}'), values are data DFs

        Returns:
            same format as input, but all DFs have
                - the same start timestamp and duration
                - resampled to the desired sampling rate
                - fill NULL at missing datapoints
        """
        data_dfs = {view: df.sort(by='timestamp(ms)') for view, df in data_dfs.items()}

        max_interval = {key: self.max_interval[key.split('_')[0]] for key in data_dfs.keys()}
        target_intervals = {key: int(1 / self.sampling_rates[key.split('_')[0]]) for key in data_dfs.keys()}

        session_start_ts = min(df.item(0, 'timestamp(ms)') for df in data_dfs.values())
        session_end_ts = max(df.item(-1, 'timestamp(ms)') for df in data_dfs.values()) + 1

        result_dfs = {
            view: pl.DataFrame({'timestamp(ms)': np.arange(session_start_ts, session_end_ts, target_intervals[view])})
            .set_sorted('timestamp(ms)')
            .join_asof(data_dfs[view], on='timestamp(ms)', strategy='nearest', tolerance=max_interval[view])
            for view in data_dfs
        }
        return result_dfs

    @staticmethod
    def concat_sensors_to_modal(data_dict: dict) -> dict:
        """
        Concat DFs of the same sensor type (same modal);
        For example: concat [inertia_1, inertia_155] => inertia

        Args:
            data_dict: dict[sensor] = DF; key has the format: '{sensor type}_{sensor id}'

        Returns:
            a dict with format: dict[modal] = DF
        """
        dfs = defaultdict(list)
        # for each sensor
        for sensor, df in data_dict.items():
            # remove sensor ID, keep sensor type
            sensor = sensor.split('_')[0]

            # only keep ts column for 1 DF of each sensor to avoid duplicate column name during concatenation
            if len(dfs[sensor]):
                df = df.select(pl.exclude('timestamp(ms)', 'frame_index'))
            dfs[sensor].append(df)

        # concat DFs with the same sensor type, add label column
        dfs = {sensor: pl.concat(list_dfs, how='horizontal') for sensor, list_dfs in dfs.items()}
        return dfs

    def assign_label(self, ske_df: pl.DataFrame, acc_df: pl.DataFrame, anno_df: pl.DataFrame) -> tuple:
        """
        Add a 'label' column
        Args:
            ske_df: skeleton DF
            acc_df: accelerometer DF
            anno_df: annotation DF for this session; filtered from self.anno_df

        Returns:
            2 DFs (ske, acc) but with `label` column
        """
        start_ts_diff = abs(ske_df.item(0, 'timestamp(ms)') - acc_df.item(0, 'timestamp(ms)'))
        end_ts_diff = abs(ske_df.item(-1, 'timestamp(ms)') - acc_df.item(-1, 'timestamp(ms)'))
        assert \
            (start_ts_diff <= self.max_interval[CMDFallConst.MODAL_SKELETON]) and \
            (end_ts_diff <= self.max_interval[CMDFallConst.MODAL_SKELETON]), \
            'Accelerometer and skeleton DFs must be synchronised.'

        # fill label column with 0 (unknown class)
        ske_df = ske_df.with_columns(pl.Series(name='label', values=np.zeros(len(ske_df), dtype=int)))

        # fill labels into column of ske DF
        for anno_row in anno_df.iter_rows(named=True):
            ske_df = ske_df.with_columns(
                label=pl.when(
                    pl.col('frame_index').is_between(anno_row['start_frame'], anno_row['stop_frame'])
                ).then(anno_row['action_id']).otherwise(pl.col('label'))
            )

        # fill label into acc DF
        acc_df = acc_df.join_asof(ske_df.select('timestamp(ms)', 'label'), on='timestamp(ms)', strategy='nearest')
        return ske_df, acc_df

    def run(self):
        logger.info('Scanning for sessions...')
        session_files = self.scan_data_files()
        logger.info(f'Found {len(session_files)} sessions in total')

        skipped_sessions = 0
        skipped_files = 0
        written_files = 0
        # for each session
        for _, session_row in session_files.iterrows():
            # get session info
            session_id, subject, sensor_id = self.get_info_from_session_file(session_row.iat[0])
            session_info = f'S{session_id}P{subject}'

            # check if already run before
            if os.path.isfile(self.get_output_file_path(CMDFallConst.MODAL_INERTIA, subject, session_info)) \
                    and os.path.isfile(self.get_output_file_path(CMDFallConst.MODAL_SKELETON, subject, session_info)):
                logger.info(f'Skipping session {session_info} because already run before')
                skipped_sessions += 1
                continue
            logger.info(f'Starting session {session_info}')

            # read all data DFs of session
            # key: modal; value: whole original session DF
            data_dfs = {
                sensor: self.read_accelerometer_df_file(data_file) if sensor.startswith(CMDFallConst.MODAL_INERTIA)
                else self.read_skeleton_df_file(data_file)
                for sensor, data_file in session_row.items()
            }

            # standardise timestamps of all DFs
            data_dfs = self.resample_session_dfs(data_dfs)

            # get annotation DF
            session_anno_df = self.anno_df.filter(pl.col('setup_id') == session_id)

            # concat sensors into modals
            data_dfs = self.concat_sensors_to_modal(data_dfs)

            # add label column and save each segment
            for i, segment in enumerate(data_segments):
                segment_info = f'{session_info}_{i}' if i < len(data_segments) - 1 else session_info
                inertial_df = segment[CMDFallConst.MODAL_INERTIA]
                skeleton_df = segment[CMDFallConst.MODAL_SKELETON]

                # add label
                skeleton_df, inertial_df = self.assign_label(skeleton_df, inertial_df, session_anno_df)
                # write files
                written = self.write_output_parquet(inertial_df, CMDFallConst.MODAL_INERTIA, subject, segment_info)
                written_files += int(written)
                skipped_files += int(not written)
                written = self.write_output_parquet(skeleton_df, CMDFallConst.MODAL_SKELETON, subject, segment_info)
                written_files += int(written)
                skipped_files += int(not written)
        logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped, '
                    f'{skipped_files} file(s) skipped')
        self.export_label_list(CMDFallConst.LABEL_DICT)


class CMDFallNpyWindow(NpyWindowFormatter):
    pass


if __name__ == '__main__':
    parquet_dir = '/mnt/data_partition/downloads/parquet_dataset/cmdfallfull'
    inertial_freq = 50
    skeletal_freq = 20

    CMDFallParquet(
        raw_folder='/mnt/data_partition/downloads/CMDFall',
        destination_folder=parquet_dir,
        sampling_rates={CMDFallConst.MODAL_INERTIA: inertial_freq,
                        CMDFallConst.MODAL_SKELETON: skeletal_freq},
        min_length_segment=4
    ).run()

    # window_size_sec = 3
    # step_size_sec = 1.5
    # min_step_size_sec = 0.5
    # CMDFall = CMDFallNpyWindow(
    #     parquet_root_dir=parquet_dir,
    #     window_size_sec=window_size_sec,
    #     step_size_sec=step_size_sec,
    #     min_step_size_sec=min_step_size_sec,
    #     max_short_window=3,
    #     modal_cols={
    #         CMDFallConst.MODAL_INERTIA: {
    #             'waist_acc': ['waist_acc_x(m/s^2)', 'waist_acc_y(m/s^2)', 'waist_acc_z(m/s^2)'],
    #             'wrist_acc': ['wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)']
    #         },
    #         CMDFallConst.MODAL_SKELETON: {
    #             'skel3': [c.format(kinect_id=3) for c in CMDFallConst.SELECTED_SKELETON_COLS]
    #         }
    #     }
    # ).run()
