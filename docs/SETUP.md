
# Setup Guide

This guide explains how to set up the Music Emotion Recognition project on a new Windows computer, starting with only Visual Studio Code installed.

## 1. Prerequisites

### 1.1 Install Visual Studio Code

Download: https://code.visualstudio.com/

Install Visual Studio Code if it is not already installed.

### 1.2 Install Git

Download: https://git-scm.com/downloads/win

Install Git for Windows.

After installation, open a new PowerShell terminal and verify:

```powershell
git --version
```

Git should display its installed version.

### 1.3 Install Python 3.12.10

This project uses Python 3.12.10 (64-bit).

Download:
https://www.python.org/downloads/release/python-31210/

Download and run the Windows 64-bit installer.

Python 3.12 can coexist with other Python versions installed on the computer.

After installation, open a new PowerShell terminal and verify:

```powershell
py -3.12 --version
```

Expected output:

```text
Python 3.12.10
```

Note: `py --version` may display a different installed Python version. Always use `py -3.12` when creating this project's virtual environment.

## 2. Clone the Repository

Open PowerShell and navigate to the folder where you want to store the project.

For example:

```powershell
cd $HOME\Documents
```

Clone the repository:

```powershell
git clone https://github.com/dianaG26/MusicEmotionRecognition.git
```

The repository is private, so Git may ask you to authenticate with a GitHub account that has access.

Navigate into the project:

```powershell
cd MusicEmotionRecognition
```

Open the project in Visual Studio Code:

```powershell
code .
```

If the `code` command is unavailable, open VS Code and select File > Open Folder.

Important: On a new computer, clone the repository only once. Later, open the existing local folder instead of cloning it again.

## 3. Python Virtual Environment

This project uses a dedicated Python virtual environment to keep its dependencies separate from other projects.

### 3.1 Create the virtual environment

Open a PowerShell terminal inside the MusicEmotionRecognition project directory.

Create the virtual environment using Python 3.12:

```powershell
py -3.12 -m venv .venv
```

This command only needs to be executed once when setting up the project on a new computer.

The `.venv` folder is excluded from Git using `.gitignore`.

### 3.2 Activate the virtual environment manually

Whenever you open a new terminal and want to work with Python in this project, run:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should display something similar to:

```text
(.venv) PS C:\path\to\MusicEmotionRecognition>
```


### 3.3 Verify the Python version

With the virtual environment activated, run:

```powershell
python --version
```

Expected output:

```text
Python 3.12.10
```

### 3.4 Install project dependencies

Once the project's `requirements.txt` file has been created, install the dependencies using:

```powershell
python -m pip install -r requirements.txt
```

The dependencies will be installed inside `.venv`, without changing the Python environments of other projects.

### 3.5 Deactivate the virtual environment

To deactivate the environment in the current terminal, run:

```powershell
deactivate
```

### 3.6 Important notes

- Create `.venv` only once per local project setup.
- Activate `.venv` manually when opening a new terminal to work with Python.
- There is no need to recreate `.venv` every time VS Code is opened.
- Git commands do not require the Python virtual environment to be active.
- On a different computer, create a new `.venv` locally.
- Do not upload `.venv` to GitHub.

## 4. Dataset

Planned dataset: MERGE Audio Balanced.

Official source:
https://zenodo.org/records/13939205

Dataset download, verification, folder structure and preprocessing instructions will be added after the dataset has been inspected.

The dataset audio files will not be committed to GitHub.

## 5. Model Training

See [TRAINING.md](TRAINING.md).

Exact training commands and experiment configurations will be documented after the models are implemented.

## 6. Running the Application

Backend and frontend setup and execution instructions will be added as the application is developed.
