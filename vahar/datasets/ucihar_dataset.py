import numpy as np
import pandas as pd
from loguru import logger

if __name__ == '__main__':
    from vahar.datasets.base_classes import NpyWindowFormatter
    from vahar.constant import G_TO_MS2
else:
    from .base_classes import NpyWindowFormatter
    from ..constant import G_TO_MS2


class UCIHARConst:
    ACC_FEATURES = [
        'total_acc_x',
        'total_acc_y',
        'total_acc_z',
    ]
    GYRO_FEATURES = [
        'body_gyro_x',
        'body_gyro_y',
        'body_gyro_z'
    ]


class UCIHARNpyWindow(NpyWindowFormatter):
    def __init__(self, raw_root_dir: str, modal_cols: dict = None):
        """
        This class reads raw data of the UCI-HAR dataset, return as numpy array.

        Args:
            raw_root_dir: path to dataset folder
            modal_cols: a 2-level dict;
                1st level key is any string (will be ignored, only to keep uniform format with the super class),
                2nd level key is sub-modal names (any new name),
                2nd level value is a list of feature names, if None, use all features in UCIHARConst.
                Example (also default value)
                    {
                        'inertia': {
                            'acc': ['total_acc_x', 'total_acc_y', 'total_acc_z'],
                            'gyro': ['body_gyro_x', 'body_gyro_y', 'body_gyro_z']
                        }
                    }
        """
        if modal_cols is None:
            modal_cols = {'inertia': {
                'acc': UCIHARConst.ACC_FEATURES,
                'gyro': UCIHARConst.GYRO_FEATURES
            }}
        else:
            assert len(modal_cols) == 1, 'Only accept 1 first level key.'
            for parquet_modal, sub_modal_dict in modal_cols.items():
                if sub_modal_dict is None:
                    modal_cols[parquet_modal] = {parquet_modal: UCIHARConst.ACC_FEATURES + UCIHARConst.GYRO_FEATURES}

        super().__init__(parquet_root_dir=raw_root_dir, window_size_sec=0, step_size_sec=0, modal_cols={})
        self.modal_cols = list(modal_cols.values())[0]

    def load_set(self, set_name: str) -> pd.DataFrame:
        """
        Load either train or test set

        Args:
            set_name: train|test

        Returns:
            a DF, each row is a data window, columns are:
                - feature name: numpy array, shape [window length, channel]
                - 'label': label int
                - 'subject': subject string
        """
        result = {}
        # read data
        for submodal, feature_names in self.modal_cols.items():
            if feature_names is None:
                feature_names = UCIHARConst.ACC_FEATURES + UCIHARConst.GYRO_FEATURES
                self.modal_cols[submodal] = feature_names

            result[submodal] = np.stack([
                np.loadtxt(f'{self.parquet_root_dir}/{set_name}/Inertial Signals/{feature}_{set_name}.txt') * G_TO_MS2
                if '_acc_' in feature else
                np.loadtxt(f'{self.parquet_root_dir}/{set_name}/Inertial Signals/{feature}_{set_name}.txt')
                for feature in feature_names
            ], axis=2)
            result[submodal] = list(result[submodal])

        # read label
        result['label'] = np.loadtxt(f'{self.parquet_root_dir}/{set_name}/y_{set_name}.txt', dtype=int)

        # read subject
        with open(f'{self.parquet_root_dir}/{set_name}/subject_{set_name}.txt', 'r') as F:
            subject = F.read()
        subject = subject.strip().split()
        result['subject'] = [f'{set_name}_{s}' for s in subject]
        result = pd.DataFrame(result)

        logger.info(f'Loaded {len(result)} windows for {set_name} set.')
        return result

    def run(self) -> pd.DataFrame:
        """
        Main processing method

        Returns:
            a DF, each row is a session, columns are:
                - 'subject': string format <train|test>_<subject ID>
                - '<modality 1>': array shape [num window, window length, features]
                - '<modality 2>': ...
                - 'label': array shape [num window]
        """
        # load data
        df = pd.concat([self.load_set('train'), self.load_set('test')])
        if self.verbose:
            for submodal, feature_names in self.modal_cols.items():
                logger.info(f'Sub-modal: {submodal}; {len(feature_names)} cols: {feature_names}')

        # split "session" by subjects
        def apply_func(group):
            new_row = {}
            for col in group.columns:
                if col == 'label':
                    new_row['label'] = group['label'].to_numpy()
                elif col != 'subject':
                    new_row[col] = np.stack(group[col].tolist())
            return pd.Series(new_row)

        df = df.groupby('subject')
        df = df.apply(apply_func)
        df = df.reset_index(drop=False)
        return df


if __name__ == '__main__':
    ucihar = UCIHARNpyWindow(
        raw_root_dir='/mnt/data_drive/projects/raw datasets/UCI HAR Dataset/',
        modal_cols={
            'any name <(")': {
                'acc': UCIHARConst.ACC_FEATURES,
                # 'gyro': UCIHARConst.GYRO_FEATURES
            }
        }
    )
    dataset = ucihar.run()
    row0 = dataset.iloc[1].tolist()
    _ = 1
