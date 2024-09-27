# NNSPIKE

A LEGO SPIKE Robot using Neural Network

## Abstract

The project integrates a LEGO SPIKE Prime Hub and a Raspberry Pi to implement a line follower using a image model which was inspired by Nvidia in 2016[1]. The model predicts the x-coordinate of the line that is going to follow. The output power is then derived from the offset to the centroid of the robot with regard to the following line. For the straight lines and the curves, the training data were collected by an OpenCV function that calculates the biggest contour's moments in the image, while the other training data (at the intersection) were labeled manually. By feeding the model with about _22,000_ images and training the model in _1.0_ hours, the loss value eventually converged to around _0.001_. The project is used for a robot contest[2] in Japan.

## Environment

- OS: Raspberry Pi OS Bookworm 64-bit
- Python: 3.11.2
- pytorch: 2.3.1

> Note: A 64-bit Raspberry Pi is required because `pytorch` doesn't work on a 32-bit system.

## Gettting Started

1.  Install dependencies on Host PC

        pip install -r requirements.txt

2.  Upload `spike/main.py` to LEGO SPIKE Prime Hub, the VSCode plugin **LEGO SPIKE Prime / MINDSTORMS Robot Inventor Extension** is required. For more details, see [here](https://marketplace.visualstudio.com/items?itemName=PeterStaev.lego-spikeprime-mindstorms-vscode)

    > Note: Sometimes, the hub might be running a cached version of the program. Try restarting the hub to clear any cached programs and upload again.

3.  Connect to Raspberry Pi to open the remote terminal by a ssh client such as [PuTTY](https://www.putty.org/) or the VSCode [Remote Development Plugin](https://code.visualstudio.com/docs/remote/ssh). Install the dependencies on Raspberry Pi by

        pip install -r requirements-raspi.txt

4.  Select the corresponing slot on SPIKE and press the button to launch the script of `spike/main.py`.
5.  Run the command `python run_slient.py` on Raspberry Pi to starting the lego spike robot.

## Project Structure

```
.
└── nnspike/
    ├── spike/
    │   └── main.py                 # Program in LEGO Spike Prime for interacting with Raspberry Pi
    ├── nnspike/
    │   ├── data/
    │   │   ├── aug.py              # Data augmentation
    │   │   ├── dataset.py          # Pytorch dataset definition
    │   │   └── preprocess.py       # Data preprocessing(labeling, balancing, etc.)
    │   ├── models/
    │   │   ├── nvidia.py           # Model implementation [1]
    │   │   └── mobilenetv2.py      # Model implementation(Modified for regression task) [3]
    │   ├── unit/
    │   │   └── etrobot.py          # Interface for controlling the Robot
    │   ├── utils/
    │   │   ├── follower.py         # Line follower based on opencv
    │   │   ├── image.py            # Methods related to computer vision
    │   │   └── pid.py              # PID Controller implementation
    │   └── constant.py             # Define global constant
    ├── storage/                    # Folder for storing training data and models
    ├── scripts/                    # Scripts for data labeling, inspection, evaluation, etc.
    ├── run_slient.py               # Starting Robot
    ├── run.py                      # Starting Robot with sending image data to host PC
    ├── requirements.txt            # Win/Unix dependencies for performing tasks of model training & data augmentation
    ├── requirements-reapi.txt      # Raspberry Pi dependencies for running the Robot
    └── README.md                   # This file
```

## Training Process

A full training process is described in the jupyter notebook `./notebooks/train.ipynb`.

### Data Collection

The training data were collected by using the OpenCV `VideoCapture` object to capture a video from the Pi Camera Module, and then extracting frames of the captured video and saving the frame files to a specific folder. The function `extract_video_frames()` will save frame files following the format of `frame_{num}.png`, where `num` is the frame number. The full code sample is shown below

```python
    from nnspike.utils import extract_video_frames

    timestamp = "20240827201028"
    extract_video_frames(f"./storage/videos/{timestamp}_picamera.avi", f"./storage/frames/{timestamp}/")
```

### Data Labeling

1. The function `create_label_df()` will return a pandas dataframe for a folder that stores the frame files.
2. By default, the dataframe's records are ordered by its filename which is not convenient for inspection. You can feed the dataframe to the function `sort_by_frames_number()` to sort the dataframe by its frame number.
3. Besides the frames in the intersection, most data can be labeled automatically by calculating the image _moments_ through the OpenCV library. Setting the appropriate ROI(Region of Interest) and feeding the dataframe to the function `label_dataset_by_opencv()` will fill the value of the column `mx`, which represents the x-coordinates of the moments for the lines that need to be followed.
4. Save the dataframe to a csv file by `df.to_csv(f"./storage/frames/{timestamp}_label.csv")` where the variable _timestamp_ should be the same as the frames folder name.
5. The script `./scripts/labeler.py` is for the other records that need to be labeled manually. You can dynamically update the values in the csv file and view the result.

```python
    from nnspike.data import label_dataset_by_opencv, sort_by_frames_number, create_label_df
    from nnspike.constant import ROI_OPENCV

    df = create_label_df(f"../storage/frames_goal/{timestamp}/*.png", course="left", worker="opencv")
    df = sort_by_frames_number(df)
    df = label_dataset_by_opencv(df=df, interval=1, line_types=[0], use=True, worker="opencv", roi=ROI_OPENCV)
    df.to_csv(f"../storage/frames/{timestamp}_label.csv", index=False)
```

### Data Augmentation & Data Balancing

The image data augmentation method `ShiftScaleRotate` from Albumentations[4] has been used in the project. About half of the data was augmented by the library. The following code snippet augments a set of data and exports those data to a specific folder. The function also returns a dataframe that contains the augmented frames' path together with other necessary columns.

```python
    from nnspike.data import augment_dataset

    aug_df = augment_dataset(df=df, intervals=[1, 1.2], line_types=[0,1,2,3], p=1, export_path="../storage/frames/aug")
```

The function `balance_dataset()` simply removes the samples if the number of the corresponding column's samples exceeds the specified limit. The following code snippet creates a histogram based on the column `line_type` with 30 bins, and randomly removes samples from the bins to reduce the number of data to below 2000.

```python
    from nnspike.data import balance_dataset

    df = balance_dataset(df, 'line_type', 2000, 30)
```

### DataFrame Columns Description

The dataframe of training data contains the image path, the offset to x-coordinate, etc. The brief description for each column is summarized as the following table.  
| Column | Type | Description |
| --- | --- | --- |
| _image_path_ | `string` | Relative path of the image |
| _mx_ | `float` | Image moments calculated by OpenCV |
| _predicted_x_ | `float` | Offset of x-coordinate predicted by an existing model |
| _adjusted_x_ | `float` | Offset of x-coordinate labelled manually |
| _line_type_ | `float` | Indicates line type, where 1: straight; 2: curve; 3: intersection; 4: special(Model failed to predict) |
| _course_ | `string` | The contest contains 'left' and 'right' courses which need to train the model separately |
| _interval_ | `float` | Indicates the interval, the contest contains a intersection which requires the robot to go in a different direction for a similar image |
| _use_ | `boolean` | Whether the data is used to training the model |
| _worker_ | `string` | The labeller name |

> - By _horizontally flipping_ the image, the data can be used to train a model for the opposite course.
> - The **image_path** is the _X_ value, while the _Y_ value will be mapped to the order of **adjusted_x** > **predicted_x** > **mx** if > the column is not null.
> - Because the CNN architecture does not have the ability to retain the previous memories, so it is necessary to divide the contest course into multiple intervals to ensure that each interval does not contain any similar images that require the robot to turn in a different direction. In the contest [2], the value of **interval** includes `1.0`, `1.2`, `2`, `2.3` and `3`, where the data of `1.2` and `2.3` can be both trained for **interval** `1.0`, `2.0` and `2.0`, `3.0` represents the period to switch models.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## References

[1]. [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)  
[2]. [ET Robocon Github Repository](https://github.com/ETrobocon)  
[3]. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  
[4]. [Albumentations: Fast and flexible image augmentation library](https://github.com/albumentations-team/albumentations)
