import os
import re
from collections import defaultdict
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
    from vahar.modal_sync import split_interrupted_dfs
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from ..constant import G_TO_MS2
    from ..modal_sync import split_interrupted_dfs


class CMDFallConst:
    # modal names
    MODAL_INERTIA = 'inertia'
    MODAL_SKELETON = 'skeleton'

    # raw sampling rates in Hertz
    RAW_INERTIA_FREQ = 50
    RAW_KINECT_FPS = 20

    ACCELEROMETER_POSITION = {1: 'wrist', 155: 'waist'}

    JOINTS_LIST = [
        'hipCenter', 'spine', 'shoulderCenter', 'head',
        'leftShoulder', 'leftElbow', 'leftWrist', 'leftHand',
        'rightShoulder', 'rightElbow', 'rightWrist', 'rightHand',
        'leftHip', 'leftKnee', 'leftAnkle', 'leftFoot',
        'rightHip', 'rightKnee', 'rightAnkle', 'rightFoot'
    ]
    SELECTED_JOINT_LIST = JOINTS_LIST.copy()
    # [
    #     'hipCenter', 'shoulderCenter',
    #     'leftElbow', 'leftWrist',
    #     'rightElbow', 'rightWrist',
    #     'leftKnee', 'leftAnkle',
    #     'rightKnee', 'rightAnkle',
    # ]

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
            3: np.array([0.0277538, 0.9024955, -0.42962335, 1.630657])[[0, 2, 1, 3]]
        }
        cls.SKELETON_ROT_MAT = {}
        for kinect_id, floor_eq in floor_eqs.items():
            norm2 = np.linalg.norm(floor_eq[:3])
            rot_angle = np.arccos(floor_eq[2] / norm2)
            rot_axis = np.cross(floor_eq[:3], [0, 0, 1]) / norm2
            cls.SKELETON_ROT_MAT[kinect_id] = axangle2mat(rot_axis, rot_angle)


CMDFallConst.define_att()


class CMDFallParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 min_length_segment: float = 10,
                 use_accelerometer: list = [1, 155]):
        """
        Class for processing CMDFall dataset.
        Use only Inertial sensors and Camera 3.
        Labels:
        - 0: unknown label
        - from 1 to 20: as in the raw dataset

        Args:
            min_length_segment: only write segments longer than this threshold (unit: sec)
            use_accelerometer: inertial sensor IDs
        """
        assert len(use_accelerometer), 'No accelerometer is used?'
        assert len(set(use_accelerometer) - {1, 155}) == 0, 'Invalid inertial sensor ID. Allowed: [1, 155]'
        super().__init__(raw_folder, destination_folder, sampling_rates)

        self.min_length_segment = min_length_segment * 1000
        self.use_accelerometer = use_accelerometer
        self.use_kinect = [3]

        # if actual interval > expected interval * this coef; it's considered an interruption and DF will be split
        max_interval_coef = 4
        # maximum intervals in millisecond
        self.max_interval = {
            CMDFallConst.MODAL_INERTIA: 1000 / CMDFallConst.RAW_INERTIA_FREQ * max_interval_coef,
            CMDFallConst.MODAL_SKELETON: 1000 / CMDFallConst.RAW_KINECT_FPS * max_interval_coef
        }

        # read annotation file
        anno_df = pl.read_csv(f'{raw_folder}/annotation.csv')
        self.anno_df = anno_df.filter(pl.col('kinect_id') == self.use_kinect[0])

        self.label_dict = {
            0: 'unknown', 1: 'walk', 2: 'run_slowly', 3: 'static_jump', 4: 'move_hand_and_leg', 5: 'left_hand_pick_up',
            6: 'right_hand_pick_up', 7: 'stagger', 8: 'front_fall', 9: 'back_fall', 10: 'left_fall', 11: 'right_fall',
            12: 'crawl', 13: 'sit_on_chair_then_stand_up', 14: 'move_chair', 15: 'sit_on_chair_then_fall_left',
            16: 'sit_on_chair_then_fall_right', 17: 'sit_on_bed_and_stand_up', 18: 'lie_on_bed_and_sit_up',
            19: 'lie_on_bed_and_fall_left', 20: 'lie_on_bed_and_fall_right'
        }

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

        # handle erroneous timestamps (negative intervals in timestamp column, but the data are fine)
        # find negative intervals
        ts = df.get_column('timestamp(ms)').to_numpy().copy()
        intervals = np.diff(ts)
        standard_interval = np.median(intervals)
        neg_idx = np.nonzero(intervals < 0)[0]
        # shift timestamps forward
        for idx in neg_idx:
            ts[idx + 1:] = ts[idx + 1:] - intervals[idx] + standard_interval
        # assign corrected timestamps
        df = df.with_columns(pl.lit(ts, dtype=pl.Int64).alias('timestamp(ms)'))

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

    def split_session_to_segments(self, data_dfs: dict) -> list:
        """
        Split dataframes of a session into uninterrupted segments

        Args:
            data_dfs: a dict with keys are sensor (format: '{sensor type}_{sensor id}'), values are data DFs

        Returns:
            list of uninterrupted segments; each one is a dict with keys are modal names, values are DFs
        """
        # split interrupted signals into uninterrupted sub-sessions
        max_interval = {key: self.max_interval[key.split('_')[0]] for key in data_dfs.keys()}
        sampling_rates = {key: self.sampling_rates[key.split('_')[0]] for key in data_dfs.keys()}
        segments = split_interrupted_dfs(
            data_dfs, max_interval=max_interval, min_length_segment=self.min_length_segment,
            sampling_rates=sampling_rates
        )
        return segments

    def concat_sensors_to_modal(self, data_dict: dict) -> dict:
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

        skip_session_files = 0
        write_segment_files = 0
        # for each session
        for _, session_row in session_files.iterrows():
            # get session info
            session_id, subject, sensor_id = self.get_info_from_session_file(session_row.iat[0])
            session_info = f'S{session_id}P{subject}'

            # check if already run before
            if os.path.isfile(self.get_output_file_path(CMDFallConst.MODAL_INERTIA, subject, session_info)) \
                    and os.path.isfile(self.get_output_file_path(CMDFallConst.MODAL_SKELETON, subject, session_info)):
                logger.info(f'Skipping session {session_info} because already run before')
                skip_session_files += 1
                continue
            logger.info(f'Starting session {session_info}')

            # read all data DFs of session
            # key: modal; value: whole original session DF
            data_dfs = {
                sensor: self.read_accelerometer_df_file(data_file) if sensor.startswith(CMDFallConst.MODAL_INERTIA)
                else self.read_skeleton_df_file(data_file)
                for sensor, data_file in session_row.items()
            }
            # split session into uninterrupted segments
            data_segments = self.split_session_to_segments(data_dfs)
            # concat sensors into modals
            data_segments = [self.concat_sensors_to_modal(seg) for seg in data_segments]

            # get annotation DF
            session_anno_df = self.anno_df.filter(pl.col('setup_id') == session_id)

            # add label column and save each segment
            for i, segment in enumerate(data_segments):
                this_session_info = f'{session_info}_{i}' if i < len(data_segments) - 1 else session_info
                inertial_df = segment[CMDFallConst.MODAL_INERTIA]
                skeleton_df = segment[CMDFallConst.MODAL_SKELETON]

                # add label
                skeleton_df, inertial_df = self.assign_label(skeleton_df, inertial_df, session_anno_df)
                # write files
                if self.write_output_parquet(inertial_df, CMDFallConst.MODAL_INERTIA, subject, this_session_info):
                    write_segment_files += 1
                if self.write_output_parquet(skeleton_df, CMDFallConst.MODAL_SKELETON, subject, this_session_info):
                    write_segment_files += 1
        logger.info(f'{write_segment_files} file(s) written, {skip_session_files} session(s) skipped')
        self.export_label_list()


class CMDFallNpyWindow(NpyWindowFormatter):
    def run(self) -> pd.DataFrame:
        # get list of parquet files
        parquet_sessions = self.get_parquet_file_list()

        result = []
        # for each session
        for parquet_session in parquet_sessions.iter_rows(named=True):
            # get session info
            modal, subject, session_id = self.get_parquet_session_info(list(parquet_session.values())[0])

            session_result = self.parquet_to_windows(parquet_session=parquet_session, subject=subject)

            result.append(session_result)
        result = pd.DataFrame(result)
        return result


if __name__ == '__main__':
    parquet_dir = '/mnt/data_partition/UCD/dataset_processed/CMDFall20'
    inertial_freq = 50
    skeletal_freq = 20
    window_size_sec = 3
    step_size_sec = 1.5
    min_step_size_sec = 0.5

    CMDFallParquet(
        raw_folder='/mnt/data_partition/downloads/CMDFall',
        destination_folder=parquet_dir,
        sampling_rates={CMDFallConst.MODAL_INERTIA: inertial_freq,
                        CMDFallConst.MODAL_SKELETON: skeletal_freq}
    ).run()

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
