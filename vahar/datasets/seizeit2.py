import os
from collections import defaultdict
import numpy as np
import pyedflib
from loguru import logger
import polars as pl
from glob import glob
import orjson
import re
from my_py_utils.my_py_utils.pl_dataframe import resample_numeric_df

if __name__ == '__main__':
    from vahar.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter


class SeizeIT2Const:
    # modal names; currently only support accelerometer
    MODAL_ELECTRO = 'electro'
    MODAL_INERTIA = 'inertia'

    CHANNEL_NAME_MAPPING = {
        'BTEleft SD': 'eegLeft',
        'BTEright SD': 'eegRight',
        'CROSStop SD': 'eegCross',
        'ECG SD': 'ecg',
        'EMG SD': 'emg',
        'EEG SD ACC X': 'headAccX',
        'EEG SD ACC Y': 'headAccY',
        'EEG SD ACC Z': 'headAccZ',
        'EEG SD GYR A': 'headGyrX',
        'EEG SD GYR B': 'headGyrY',
        'EEG SD GYR C': 'headGyrZ',
        'ECGEMG SD ACC X': 'torsoAccX',
        'ECGEMG SD ACC Y': 'torsoAccY',
        'ECGEMG SD ACC Z': 'torsoAccZ',
        'ECGEMG SD GYR A': 'torsoGyrX',
        'ECGEMG SD GYR B': 'torsoGyrY',
        'ECGEMG SD GYR C': 'torsoGyrZ',
    }


class SeizeIT2Parquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict = None,
                 read_modals=('eeg', 'ecg', 'emg', 'mov')):
        """
        Class for SeizeIT2 dataset.

        Args:
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            sampling_rates: a dict containing sampling rates of each modal to resample by linear interpolation.
                - key: modal name
                - value: sampling rate (unit: Hz)
            read_modals: raw modalities to read
        """
        if sampling_rates is None:
            sampling_rates = {SeizeIT2Const.MODAL_ELECTRO: 256, SeizeIT2Const.MODAL_INERTIA: 25}
        else:
            sampling_rates = sampling_rates

        super().__init__(raw_folder, destination_folder, sampling_rates)
        self.read_modals = read_modals

        # read label list
        with open(f'{self.raw_folder}/events.json', 'rb') as F:
            events = orjson.loads(F.read())
        label_list = ['null'] + [l for l in events['eventType']['Levels']
                                 if l not in {'bckg', 'impd'}]
        self.label_dict = dict(zip(range(len(label_list)), label_list))
        self.label2index = dict(zip(label_list, range(len(label_list))))

        lateralization_list = list(events['lateralization']['Levels'])
        self.lateral2index = dict(zip(
            lateralization_list,
            [l[0].upper() for l in lateralization_list]
        ))

    @staticmethod
    def get_info_from_file_path(path: str) -> tuple[str, str]:
        """
        Get session info from file path.
        Example: sub-055/ses-01/eeg/sub-055_ses-01_task-szMonitoring_run-01_eeg.edf

        Args:
            path: path to a raw data csv file

        Returns:
            a tuple of (subject ID, run ID). Example: ('sub-055', 'run-01')
        """
        info = re.search('(sub-\d+)_ses-01_task-szMonitoring_(run-\d+)_eeg.edf$', path)
        subject_id = info.group(1)
        run_id = info.group(2)
        return subject_id, run_id

    def read_edf_session(self, subject_id: str, run_id: str, read_modals: list = None) -> dict[str, pl.DataFrame]:
        """
        Read data of 1 session from EDF files, each of which contains data of 1 modality (eeg, ecg, emg, mov).

        Args:
            subject_id: subject ID.
            run_id: run ID.
            read_modals: modalities to read

        Returns:
            dict[modal] = dataframe
        """
        electro_data = {}
        electro_sampling_rate = None
        inertial_data = {}
        inertial_sampling_rate = None

        # load data from EDF files
        read_modals = read_modals or self.read_modals
        for modal in read_modals:
            edf_file = os.path.join(
                self.raw_folder, subject_id, 'ses-01', modal,
                '_'.join([subject_id, 'ses-01_task-szMonitoring', run_id, modal + '.edf'])
            )
            if not os.path.exists(edf_file):
                # logger.warning(f'{edf_file} does not exist.')
                continue

            # read data from file
            with pyedflib.EdfReader(edf_file) as edf:
                sampling_rates = edf.getSampleFrequencies()
                channel_names = edf.getSignalLabels()

                for i, channel_name in enumerate(channel_names):
                    if channel_name.endswith('ACC Z'):
                        continue
                    channel_name = SeizeIT2Const.CHANNEL_NAME_MAPPING[channel_name]
                    if modal == 'mov':
                        inertial_data[channel_name] = edf.readSignal(i)
                        if inertial_sampling_rate is None:
                            inertial_sampling_rate = sampling_rates[i]
                        else:
                            assert inertial_sampling_rate == sampling_rates[i]
                    else:
                        electro_data[channel_name] = edf.readSignal(i)
                        if electro_sampling_rate is None:
                            electro_sampling_rate = sampling_rates[i]
                        else:
                            assert electro_sampling_rate == sampling_rates[i]

        # add timestamp column, re-sample
        results = {}
        for modal, data, sampling_rate in [
            [SeizeIT2Const.MODAL_ELECTRO, electro_data, electro_sampling_rate],
            [SeizeIT2Const.MODAL_INERTIA, inertial_data, inertial_sampling_rate]
        ]:
            if len(data) == 0:
                continue

            data = pl.DataFrame(data)

            # convert sampling rate from Hz to sample/ms
            sampling_rate = sampling_rate / 1000

            # add timestamp column
            data = data.with_columns(pl.lit(
                (np.arange(len(data)) / sampling_rate).round(),
                dtype=pl.Int64
            ).alias('timestamp(ms)'))

            # re-sample data
            if sampling_rate != self.sampling_rates[modal]:
                data = resample_numeric_df(
                    data, timestamp_col='timestamp(ms)', new_frequency=self.sampling_rates[modal]
                )

            data = data.with_columns(pl.exclude('timestamp(ms)').cast(pl.Float32))
            results[modal] = data

        return results

    def add_label_col(self, df_dict: dict[str, pl.DataFrame], subject_id: str, run_id: str) -> dict[str, pl.DataFrame]:
        """
        Add label column to modal dataframes of the same session.

        Args:
            df_dict: dict[modal name] = df
            subject_id: subject ID.
            run_id: run ID.

        Returns:
            same format as df_dict.
        """
        df_dict = {
            modal: df.with_columns(label=0, lateralization=None)
            for modal, df in df_dict.items()
        }

        # read annotation file
        events = pl.read_csv(
            os.path.join(
                self.raw_folder, subject_id, 'ses-01', 'eeg',
                '_'.join([subject_id, 'ses-01_task-szMonitoring', run_id, 'events.tsv'])
            ),
            columns=['onset', 'duration', 'eventType', 'lateralization'],
            separator='\t',
        )
        events = events.filter(pl.col('eventType').is_in({'bckg', 'impd'}).not_())
        if len(events) == 0:
            return df_dict

        # convert event timestamps
        events = events.with_columns(end=pl.col('onset') + pl.col('duration')) \
            .with_columns(pl.col('onset', 'end') * 1000)

        # add label to the data DFs
        for event in events.iter_rows(named=True):
            label = self.label2index[event['eventType']]
            lat = self.lateral2index[event['lateralization']]
            condition = pl.when(pl.col('timestamp(ms)').is_between(event['onset'], event['end'], closed='left'))
            label_expr = condition.then(pl.lit(label)).otherwise(pl.col('label')).alias('label')
            lateral_expr = condition.then(pl.lit(lat)).otherwise(pl.col('lateralization')).alias('lateralization')

            for modal, df in df_dict.items():
                df_dict[modal] = df.with_columns(label_expr, lateral_expr)

        return df_dict

    def run(self):
        written_files = 0
        skipped_sessions = 0
        skipped_files = 0

        # for each session
        from tqdm import tqdm
        for eeg_file in tqdm(sorted(glob(os.sep.join([self.raw_folder, 'sub*', 'ses-01', 'eeg', '*.edf'])))):
            subject_id, run_id = self.get_info_from_file_path(eeg_file)
            session_id = f'{subject_id}_{run_id}'

            # check if already run before
            subject_id_int = int(subject_id.removeprefix('sub-'))
            if os.path.isfile(self.get_output_file_path(SeizeIT2Const.MODAL_INERTIA, subject_id_int, session_id)) and \
                    os.path.isfile(self.get_output_file_path(SeizeIT2Const.MODAL_ELECTRO, subject_id_int, session_id)):
                logger.info(f'Skipping session {session_id} because it has been done before.')
                skipped_sessions += 1
                continue
            logger.info(f'Starting session {session_id}')

            # read file
            df_dict = self.read_edf_session(subject_id, run_id)

            # add label column
            df_dict = self.add_label_col(df_dict, subject_id, run_id)

            # write DF file
            for modal, df in df_dict.items():
                written = self.write_output_parquet(
                    df, modal, subject_id_int, session_id,
                    nan_cols=['lateralization'],
                    check_interval=(modal == SeizeIT2Const.MODAL_INERTIA)
                )
                written_files += int(written)
                skipped_files += int(not written)

        logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped, '
                    f'{skipped_files} file(s) not written.')

        # convert labels from text to numbers
        self.export_label_list()


class SeizeIT2NpyWindow(NpyWindowFormatter):
    pass


class SeizeIT2NpzFile(SeizeIT2Parquet, NpyWindowFormatter):
    """
    This class saves processed data as numpy arrays of windows.
    """
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict = None,
                 read_modals=('eeg', 'ecg', 'emg', 'mov'),

                 window_size_sec=2, step_size_sec=2,
                 min_step_size_sec=0.5, max_short_window=float('inf'), exclude_labels=tuple(),
                 no_transition=False):
        SeizeIT2Parquet.__init__(self, raw_folder, destination_folder, sampling_rates, read_modals)
        NpyWindowFormatter.__init__(self, None, window_size_sec, step_size_sec,
                                    min_step_size_sec, max_short_window, exclude_labels, no_transition,
                                    modal_cols={SeizeIT2Const.MODAL_ELECTRO: None, SeizeIT2Const.MODAL_INERTIA: None})

        self.OUTPUT_PATTERN = '{root}/{modal}/subject_{subject}/{session}_{columns}_{num_window}.npy'
        self.label_dict = {0: 'null', 1: 'seizure'}

    @staticmethod
    def format_annotation_df(event_df: pl.DataFrame, total_duration, start_col='onset', end_col='end') -> pl.DataFrame:
        """
        Add non-event rows to a Polar DataFrame containing event start and end times.

        Args:
            event_df: polars DataFrame with start and end columns
            total_duration: total recording duration
            start_col: start timestamp column
            end_col: end timestamp column

        Returns:
            Polars DataFrame with both events and non-events,
            including a 'label' column (1 for event, 0 for non-event)
        """
        # Sort events by start time
        assert event_df.get_column(start_col).equals(event_df.get_column(start_col).sort())

        # merge events if there's overlapping
        ts = event_df.select(start_col, end_col).to_numpy().reshape(-1)
        event_overlapped = (ts != np.sort(ts)).any()
        if event_overlapped:
            # Initialize lists for merged events
            merged_starts = []
            merged_ends = []

            # Merge overlapping events
            current_start = None
            current_end = None
            for event in event_df.iter_rows(named=True):
                if current_start is None:
                    current_start = event[start_col]
                    current_end = event[end_col]
                elif event[start_col] <= current_end:
                    # Events overlap, extend current_end if necessary
                    current_end = max(current_end, event[end_col])
                else:
                    # No overlap, save current event and start new one
                    merged_starts.append(current_start)
                    merged_ends.append(current_end)
                    current_start = event[start_col]
                    current_end = event[end_col]

            # Add last event if exists
            if current_start is not None:
                merged_starts.append(current_start)
                merged_ends.append(current_end)

            # Create merged events DataFrame
            event_df = pl.DataFrame({
                start_col: merged_starts,
                end_col: merged_ends,
                'label': True
            })

        # add non-event annotation rows
        intervals = []

        # Add initial non-event if there's a gap before first event
        if event_df.item(0, start_col) > 0:
            intervals.append({start_col: 0, end_col: event_df.item(0, start_col), 'label': False})

        # Process each event and add gaps between events
        for i, event in enumerate(event_df.iter_rows(named=True)):
            # Add the current event
            intervals.append({start_col: event[start_col], end_col: event[end_col], 'label': True})

            # Add gap after current event if it's not the last one
            if i < len(event_df) - 1:
                gap_start = event[end_col]
                gap_end = event_df.item(i + 1, start_col)

                if gap_end > gap_start:
                    intervals.append({start_col: gap_start, end_col: gap_end, 'label': False})

        # Add final non-event if there's a gap after last event
        if event_df.item(-1, end_col) < total_duration:
            intervals.append({start_col: event_df.item(-1, end_col), end_col: total_duration, 'label': False})

        # Create new DataFrame from intervals
        result_df = pl.DataFrame(intervals)
        return result_df

    def split_by_label(self, df_dict: dict[str, pl.DataFrame], subject_id: str, run_id: str) -> list:
        """
        Add label column to modal dataframes of the same session.

        Args:
            df_dict: dict[modal] = df
            subject_id: subject ID.
            run_id: run ID.

        Returns:
            list of dict, each dict has the same format as the input; each dataframe has an extra column 'label'.
        """
        # read annotation file
        events = pl.read_csv(
            os.path.join(
                self.raw_folder, subject_id, 'ses-01', 'eeg',
                '_'.join([subject_id, 'ses-01_task-szMonitoring', run_id, 'events.tsv'])
            ),
            columns=['onset', 'duration', 'eventType', 'recordingDuration'],
            separator='\t',
        )
        events = events.filter(pl.col('eventType').is_in({'bckg', 'impd'}).not_())

        # return if there's no event
        if len(events) == 0:
            for modal in df_dict.keys():
                df_dict[modal] = df_dict[modal].with_columns(label=pl.lit(0))
            return [df_dict]

        # check event types
        for i, event in enumerate(events.iter_rows(named=True)):
            assert event['eventType'].startswith('sz')

        # convert event timestamps
        events = events.with_columns(end=pl.col('onset') + pl.col('duration')) \
            .with_columns(pl.exclude('eventType') * 1000)

        # add non-event rows and column 'label' to events DF
        events = self.format_annotation_df(events, total_duration=events.item(0, 'recordingDuration'))

        # split data by event
        results = []
        for i, event in enumerate(events.iter_rows(named=True)):
            data_df = {
                modal: df.filter(pl.col('timestamp(ms)').is_between(event['onset'], event['end'], closed='left')) \
                    .with_columns(label=pl.lit(event['label']))
                for modal, df in df_dict.items()
            }
            results.append(data_df)

        return results

    def write_output_npz(self, arr_dict: dict[str, np.ndarray], modal, subject_id, session_id) -> bool:
        """
        Export data to NPZ file.

        Args:
            arr_dict: a dict containing numpy arrays
            modal: modal name
            subject_id: subject ID
            session_id: session ID

        Returns:
            bool indicating success.
        """
        assert 'label' in arr_dict

        output_path = self.get_output_file_path(modal=modal, subject=subject_id, session=session_id, extension='npz')
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        np.savez_compressed(output_path, **arr_dict)
        logger.info(f'Npz written with {len(arr_dict["label"])} windows: {output_path}')
        return True

    def run(self):
        """
        Main processing method
        """
        written_files = 0
        skipped_sessions = 0
        skipped_files = 0

        # for each session
        for eeg_file in sorted(glob(os.sep.join([self.raw_folder, 'sub*', 'ses-01', 'eeg', '*.edf']))):
            subject_id, run_id = self.get_info_from_file_path(eeg_file)
            session_id = f'{subject_id}_{run_id}_window-{{n_window}}_event-{{is_event}}'

            # check if already run before
            subject_id_int = int(subject_id.removeprefix('sub-'))
            if glob(self.get_output_file_path(
                    modal='all_modal', subject=subject_id_int,
                    session=session_id.format(n_window='*', is_event='*'),
                    extension='npz'
            )):
                logger.info(f'Skipping session {session_id} because it has been done before.')
                skipped_sessions += 1
                continue

            logger.info(f'Starting session {subject_id} {run_id}')

            # read file
            data = self.read_edf_session(subject_id, run_id)

            # split session into segments of the same events (same labels),
            # so different step size can be run for each label in sliding window
            data = self.split_by_label(data, subject_id, run_id)
            has_event = int(len(data) > 1)

            # sliding window
            session_data = defaultdict(list)
            for continuous_seg in data:
                seg_label = continuous_seg[list(continuous_seg)[0]].item(0, 'label')
                seg = self.parquet_to_windows(
                    parquet_session=continuous_seg,
                    subject=subject_id_int,
                    session_label=seg_label,
                    is_short_activity=bool(seg_label)  # run short activity if this is a seizure event
                )
                for key, arr in seg.items():
                    session_data[key].append(arr)

            # concat window segments
            for key in list(session_data.keys()):
                if key == 'subject':
                    session_data.pop(key)
                    continue
                if key == 'label':
                    value = np.concatenate(session_data[key]).astype(bool)
                else:
                    value = np.concatenate(session_data[key], dtype=np.float32)
                session_data[key] = value

            # add data column names
            session_data |= {
                f'{modal}_columns': np.array([c for c in df.columns if c not in {'timestamp(ms)', 'label'}])
                for modal, df in continuous_seg.items()
            }

            # write NPZ file
            written = self.write_output_npz(
                session_data, modal='all_modal', subject_id=subject_id_int,
                session_id=session_id.format(n_window=len(session_data['label']), is_event=has_event),
            )
            written_files += int(written)
            skipped_files += int(not written)

        logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped, '
                    f'{skipped_files} file(s) not written.')

        # convert labels from text to numbers
        self.export_label_list()


if __name__ == '__main__':
    SeizeIT2NpzFile(
        raw_folder='/mnt/data_partition/downloads/ds005873-download',
        destination_folder='/mnt/data_partition/downloads/npz_dataset/seizeit2',
        read_modals=['eeg', 'ecg', 'emg', 'mov'],
        window_size_sec=5, step_size_sec=2,
        min_step_size_sec=0.5, max_short_window=float('inf'),
    ).run()
