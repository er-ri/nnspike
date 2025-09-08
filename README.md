# NNSPIKE

A LEGO SPIKE Robot using Neural Network

## Abstract

This project combines a LEGO SPIKE Prime Hub with a Raspberry Pi to create an robot. Using a neural network model inspired by NVIDIA's self-driving car research[^1], the robot predicts the path to follow based on camera input. The system processes both straight lines and curves automatically through computer vision, with additional manual labeling for complex intersections. This robot was developed for participation in a Japanese robotics competition[^2].

## Environment
- LEGO: HubOS Legacy
- OS: Raspberry Pi OS Bookworm 64-bit
- Python: 3.11.2
- pytorch: 2.3.1

> [!NOTE]
> A 64-bit Raspberry Pi is required because `pytorch` doesn't work on a 32-bit system.

## Getting Started
1.  Install dependencies on Host PC

        pip install -r requirements.txt

2.  Upload `spike/slot2.py` to LEGO SPIKE Prime Hub through [**SPIKE Legacy App**](https://education.lego.com/en-us/downloads/spike-legacy-app/software/).

> Downgrade the **HubOS** from [here](https://spikelegacy.legoeducation.com/hubdowngrade/#step-1) if you are using 3.x version. For Windows environment, the tool [Zadig](https://zadig.akeo.ie/) is also required.

3.  Connect to Raspberry Pi to open the remote terminal by a ssh client such as [PuTTY](https://www.putty.org/) or the VSCode [Remote Development Plugin](https://code.visualstudio.com/docs/remote/ssh). Install the dependencies on Raspberry Pi by

        pip install -r requirements-raspi.txt

4.  Select the corresponding slot on SPIKE and press the button to launch the script of `spike/slot_prod.py`.
5.  Run the command `python run.py --model-path "to/your/model.pt"` on Raspberry Pi to starting the lego spike robot.


> [!IMPORTANT]
> This project requires `LEGO® Education SPIKE™ Legacy App v. 2.0.10` and will not work with `LEGO® Education SPIKE™ App v. 3+`. You can:
> 1. Download the Legacy App from [here](https://education.lego.com/en-us/downloads/spike-legacy-app/software/)
> 2. Use [this site](https://spikelegacy.legoeducation.com/hubdowngrade/) to downgrade your hub to version `2.0`
> 3. For Windows system, [Zadig](https://zadig.akeo.ie/) is also required

## Project Structure

```
.
├── notebooks/                      # Jupyter notebooks for model training and experimentation
│   ├── train_classification.ipynb  # Training classification model
│   ├── train_multi_task.ipynb      # Training multi-task model
│   ├── train_regression.ipynb      # Training regression model
│   └── utils.py                    # Utility functions for notebooks
├── nnspike/
│   ├── data/
│   │   ├── aug.py                 # Data augmentation
│   │   ├── dataset.py             # Pytorch dataset definition
│   │   └── preprocess.py          # Data preprocessing(labeling, balancing, etc.)
│   ├── models/
│   │   ├── customized.py          # Customized Model implementation
│   │   ├── loss.py                # Loss function definition
│   │   └── nvidia.py              # Model implementation based on [1]
│   ├── unit/
│   │   ├── action_chain.py        # Action chain implementation
│   │   ├── etrobot.py             # Interface for controlling the Robot
│   │   ├── mode_manager.py        # Mode management for the robot
│   │   ├── spike_status.py        # SPIKE status monitoring
│   │   └── webcam_video_stream.py # Video stream handling
│   ├── utils/
│   │   ├── control.py             # Robot control utilities
│   │   ├── image.py               # Methods related to computer vision
│   │   ├── pid.py                 # PID Controller implementation
│   │   └── recorder.py            # Data recording utilities
│   └── constants.py               # Define global constants
├── scripts/                       # Scripts for data labeling, inspection, evaluation
│   ├── emulator.py                # Robot emulator
│   ├── labeler.py                 # Data labeling tool
│   └── receiver.py                # Data receiving utility
├── spike/                         # SPIKE Prime related code
│   ├── slot_prod.py               # Production slot program
│   └── slot_test.py               # Test slot program
├── storage/                       # Folder for storing training data and models
├── run.py                         # Main robot control script
├── run_manual.py                  # Manual control script
├── run_rc.py                      # Remote control script
├── requirements.txt               # Host PC dependencies
├── requirements-raspi.txt         # Raspberry Pi dependencies
└── README.md                      # This file
```

## Development

### Setup Development Environment

1. Install development dependencies:
   ```bash
   uv sync --extra dev
   ```

2. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

### Code Quality Commands

- **Lint and auto-fix issues**: `uv run ruff check --fix nnspike/`
- **Format code**: `uv run ruff format nnspike/`
- **Type checking**: `uv run mypy nnspike/`
- **Pre-commit on all files**: `uv run pre-commit run --all-files`

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## References

[^1]: [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
[^2]: [ET Robocon Github Repository](https://github.com/ETrobocon)