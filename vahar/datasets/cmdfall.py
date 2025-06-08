import os
import re
from collections import defaultdict
from glob import glob
from typing import List, Dict, Union
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from transforms3d.axangles import axangle2mat
from scipy.interpolate import interp1d

from my_py_utils.my_py_utils.string_utils import rreplace
from my_py_utils.my_py_utils.number_array import np_mode

if __name__ == '__main__':
    from vahar.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from vahar.constant import G_TO_MS2
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from ..constant import G_TO_MS2


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


def plot_timelines(df_dict: dict[str, pl.DataFrame], start_col='start', end_col='end', label_col='label'):
    """
    Plot labels in annotation dataframes as parallel timelines

    Args:
        df_dict: dict of annotation dataframes
        start_col: column name for event start time
        end_col: column name for event end time
        label_col: event label
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_labels = sorted(set(label for df in df_dict.values() for label in df[label_col]))
    num_labels = len(all_labels)

    if num_labels <= 20:
        palette = sns.color_palette("tab20", num_labels)
    else:
        palette = sns.color_palette("husl", num_labels)  # fallback for >20

    label_to_color = {label: palette[i] for i, label in enumerate(all_labels)}

    fig, ax = plt.subplots(figsize=(10, len(df_dict) * 1.5))

    yticks = []
    yticklabels = []

    for i, (df_name, df) in enumerate(df_dict.items()):
        y = i
        yticks.append(y)
        yticklabels.append(df_name)

        for row in df.iter_rows(named=True):
            color = label_to_color[row[label_col]]
            ax.barh(
                y=y,
                width=row[end_col] - row[start_col],
                left=row[start_col],
                height=0.4,
                color=color,
                edgecolor='black'
            )
            ax.text(row[start_col], y + 0.05, row[label_col], va='bottom', ha='left', fontsize=9)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time")
    ax.set_title("Parallel Timelines with Consistent Label Colors")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    # Optional: create a legend
    handles = [plt.Line2D([0], [0], color=color, lw=6) for color in label_to_color.values()]
    ax.legend(handles, label_to_color.keys(), title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def merge_annotation(anno_dfs: dict[str, pl.DataFrame], start_col='start', end_col='end',
                     label_col='label') -> pl.DataFrame:
    """
    Merge multiple annotation dataframes into a single dataframe.

    Args:
        anno_dfs: dict[view name] = polars DF. Each DataFrame has columns [label_col, start_col, end_col].
        start_col: start timestamp column name
        end_col: end timestamp column name
        label_col: label column name

    Returns:
        A merged DataFrame with columns [start_col, end_col, label_col].
    """
    if len(anno_dfs) == 1:
        return list(anno_dfs.values())[0].select(start_col, end_col, label_col)

    # get all unique time boundaries
    boundaries = set()
    for df in anno_dfs.values():
        boundaries.update(df.get_column(start_col))
        boundaries.update(df.get_column(end_col))
    time_points = sorted(boundaries)

    merged_rows = []
    previous_warm_msgs = set()
    # for each non-overlapping segments
    for i in range(len(time_points) - 1):
        segment_start = time_points[i]
        segment_end = time_points[i + 1]
        segment_labels = []

        for view, df in anno_dfs.items():
            # check if this segment fits in any event in this annotation DF
            overlapping = df.filter((pl.col(start_col) <= segment_start) & (segment_end <= pl.col(end_col)))
            seg_lbl = list(set(overlapping.get_column(label_col)))
            if len(seg_lbl) > 1:
                warn_msg = (f'Conflicting labels in the same annotation DF number {view} '
                            f'from frame {overlapping.item(0, "start_frame")} '
                            f'to {overlapping.item(-1, "stop_frame")}: {seg_lbl}')
                if warn_msg not in previous_warm_msgs:
                    logger.warning(warn_msg)
                    previous_warm_msgs.add(warn_msg)

            if len(seg_lbl) > 0:
                segment_labels += seg_lbl  # label of the segment

        # apply label merging logic
        if (len(anno_dfs) > 2) and (len(segment_labels) < 2):  # only accept a label if at least 2 views agree
            merged_label = 0
        elif len(set(segment_labels)) == 1:
            merged_label = segment_labels[0]
        else:
            # vote a label
            merged_label = np_mode(np.array(segment_labels), take_1_value=False)
            if merged_label.shape:  # if there's a tie, assign as unknown label
                merged_label = 0

        if merged_label > 0:  # if not unknown label
            if (len(merged_rows) > 0) \
                    and (merged_label == merged_rows[-1][label_col]) \
                    and (segment_start == merged_rows[-1][end_col]):
                # merge with previous segment if 2 have the same label and there's no gap between 2 segments
                merged_rows[-1][end_col] = segment_end
            else:
                merged_rows.append({start_col: segment_start, end_col: segment_end, label_col: merged_label})

    merged_rows = pl.DataFrame(merged_rows)
    # plot_timelines(anno_dfs | {'vote': merged_rows}, start_col, end_col, label_col)
    return merged_rows


class CMDFallParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 max_interval_coeff=4):
        """
        Class for processing CMDFall dataset.
        Use only Inertial sensors and Camera 3.
        Labels:
        - 0: unknown activity
        - from 1 to 20: as in the raw dataset

        Args:
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            sampling_rates: a dict containing sampling rates of each modal to resample by linear interpolation.
                - key: modal name
                - value: sampling rate (unit: Hz)
            max_interval_coeff: if actual interval > expected interval * this coeff,
                it will be considered missing datapoints and will be filled with NULL
        """
        super().__init__(raw_folder, destination_folder, sampling_rates)
        self.use_accelerometer = (1, 155)
        self.use_kinect = (1, 2, 3, 4, 5, 6, 7)

        # if actual interval > expected interval * this coeff; it's considered an interruption and DF will be split
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
        df = df.sort(by='timestamp(ms)')
        return df

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
        data_df = [s.strip().split(' ') for s in data_df]

        # remove invalid frames (a valid frame has 100 features (20 joints * (3 3D skeleton + 2 2D skeleton))
        keep_idx = [row
                    for row, frame_ske in enumerate(data_df)
                    if len(frame_ske) == 100]
        if len(keep_idx) != len(data_df):
            data_df = [data_df[i] for i in keep_idx]
        data_df = np.array(data_df, dtype=float)

        data_df = data_df.reshape([len(data_df), len(CMDFallConst.JOINTS_LIST), 5])  # shape [frame, joint, axis]
        # remove 2 RGB columns, keep 3D columns
        data_df = data_df[:, :, :3]
        # switch Y and Z
        data_df = data_df[:, :, [0, 2, 1]]
        # normalise skeleton
        data_df = CMDFallParquet.normalise_skeletons(data_df, sensor_id)

        # shape [frame, joint * axis]
        data_df = data_df.reshape([len(data_df), len(CMDFallConst.SELECTED_JOINT_LIST) * 3])
        data_df = pl.DataFrame(data_df,
                               schema=[c.format(kinect_id=sensor_id) for c in CMDFallConst.SELECTED_SKELETON_COLS])

        if len(keep_idx) != len(info_df):
            info_df = info_df[keep_idx]
        df = pl.concat([info_df, data_df], how='horizontal')

        assert df.get_column('timestamp(ms)').is_sorted()
        assert df.get_column('frame_index').is_sorted()
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

    def get_session_annotation(self, data_dfs: dict, session_id: int):
        """
        Get the annotation for one session.

        Args:
            data_dfs: a dict with keys are kinect sensor (format: 'skeleton_{sensor id}'),
                values are DFs with columns 'timestamp(ms)' and 'frame_index'
            session_id: ID of the session

        Returns:
            an annotation DF for this session, columns are 'label', 'start_ts', 'end_ts'
        """
        session_anno_df = self.anno_df.filter(pl.col('setup_id') == session_id)

        view_anno_dfs = {}
        for kinect_id, cam_anno_df in session_anno_df.group_by('kinect_id'):
            kinect_id = kinect_id[0]
            cam_anno_df = cam_anno_df.sort('start_frame')

            # validate annotation DF
            all_boundaries = cam_anno_df.select('start_frame', 'stop_frame').to_numpy().reshape(-1)
            try:
                sorted_mask = all_boundaries[:-1] <= all_boundaries[1:]
                assert np.all(sorted_mask), (f'annotation DF of session {session_id}, kinect {kinect_id} '
                                             f'is not sorted at frame {all_boundaries[:-1][~sorted_mask]}')
            except AssertionError as e:
                print(e)

            # map from frame index to timestamp
            frame2ts = data_dfs[f'skeleton_{kinect_id}']
            frame2ts = interp1d(frame2ts.get_column('frame_index'), frame2ts.get_column('timestamp(ms)'),
                                fill_value='extrapolate', assume_sorted=True)

            cam_anno_df = cam_anno_df.with_columns(
                start_ts=pl.col('start_frame').map_elements(frame2ts, return_dtype=pl.Float64).round().cast(pl.Int64),
                end_ts=pl.col('stop_frame').map_elements(frame2ts, return_dtype=pl.Float64).round().cast(pl.Int64)
            )
            view_anno_dfs[kinect_id] = cam_anno_df

        view_anno_df = merge_annotation(view_anno_dfs, start_col='start_ts', end_col='end_ts', label_col='action_id')
        view_anno_df = view_anno_df.rename({'action_id': 'label'})
        return view_anno_df

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

    @staticmethod
    def assign_label(data_dfs: dict[str, pl.DataFrame], anno_df: pl.DataFrame) -> dict:
        """
        Add a 'label' column
        Args:
            data_dfs: a dict mapping view name to data DF with a 'timestamp(ms)' column
            anno_df: annotation DF for this session; filtered from self.anno_df

        Returns:
            same format as `data_dfs` but with 'label' column added
        """
        # fill label column with 0 (unknown class)
        data_dfs = {view: df.with_columns(label=0) for view, df in data_dfs.items()}

        # for each event in the annotation DF
        for anno_row in anno_df.iter_rows(named=True):
            # for each modal DF
            for modal, df in data_dfs.items():
                data_dfs[modal] = df.with_columns(
                    label=pl.when(
                        pl.col('timestamp(ms)').is_between(anno_row['start_ts'], anno_row['end_ts'], closed='left')
                    ).then(anno_row['label']).otherwise(pl.col('label'))
                )
        return data_dfs

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
                if os.path.isfile(data_file)
            }

            # get annotation DF; use the raw data_dfs to map frame index -> timestamp
            anno_df = self.get_session_annotation(data_dfs, session_id)
            # column frame_index is no longer needed
            for view in list(data_dfs):
                if 'frame_index' in data_dfs[view]:
                    data_dfs[view] = data_dfs[view].drop('frame_index')

            # standardise timestamps of all DFs
            data_dfs = self.resample_session_dfs(data_dfs)

            # concat sensors of the same modality into the same DF (same frequency => same DF size)
            data_dfs = self.concat_sensors_to_modal(data_dfs)

            # add label to the DF
            data_dfs = self.assign_label(data_dfs, anno_df)

            # save data files
            for modal_name, modal_df in data_dfs.items():
                written = self.write_output_parquet(
                    modal_df, modal_name, subject, session_info,
                    allow_nan_cols=tuple(c for c in modal_df.columns if c not in {'label', 'timestamp(ms)'}),
                )
                written_files += int(written)
                skipped_files += int(not written)
        logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped, '
                    f'{skipped_files} file(s) skipped')
        self.export_label_list(CMDFallConst.LABEL_DICT)


class CMDFallNpyWindow(NpyWindowFormatter):
    pass


if __name__ == '__main__':
    parquet_dir = '/home/nda97531/Documents/datasets/dataset_parquet/cmdfull'
    inertial_freq = 50
    skeletal_freq = 20

    CMDFallParquet(
        raw_folder='/home/nda97531/Documents/datasets/CMDFall',
        destination_folder=parquet_dir,
        sampling_rates={CMDFallConst.MODAL_INERTIA: inertial_freq,
                        CMDFallConst.MODAL_SKELETON: skeletal_freq}
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
