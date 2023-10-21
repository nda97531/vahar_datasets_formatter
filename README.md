# VAHAR - formatter for HAR datasets

This code processes HAR timeseries datasets to the same format for easier usage. The file `example.py` shows a usage
example.

## Directory tree of the formatted dataset

`{dataset root} / {modality} / subject_{subject ID} / {session ID}.parquet`

Example: `CMDFall/inertia/subject_1/S7P1_0.parquet`

## Parquet file format

Apache parquet file written using [polars](https://www.pola.rs/).<br/>
Columns:

- `timestamp(ms)`
- `label`
- feature columns...

## Supported datasets

| Dataset                                                                                                                  | Supported modalities                                                                                                                                                                                                                                                                                                       |
|--------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [CMDFall](http://mica.edu.vn:8000/KinectData/public/)                                                                    | inertia (2 accelerometers), skeleton (normalised 3D pose from Kinect 3)                                                                                                                                                                                                                                                    |
| [FallAllD](https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living) | inertia (acc and gyro)                                                                                                                                                                                                                                                                                                     |
| [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/)           | inertia (acc, gyro)                                                                                                                                                                                                                                                                                                        |
| [SFU-IMU](https://www.frdr-dfdr.ca/repo/dataset/6998d4cd-bd13-4776-ae60-6d80221e0365)                                    | inertia (acc, gyro, and mag)                                                                                                                                                                                                                                                                                               |
| [UP-Fall](https://sites.google.com/up.edu.mx/har-up/)                                                                    | inertia (acc, gyro), skeleton (normalised 2D pose extracted from Camera2 of this dataset using [OpenPose Google Colab](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source-on-ros-docker-and-google-colab---community-based-work)) |
| [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)                          | inertia (acc and gyro)                                                                                                                                                                                                                                                                                                     |
