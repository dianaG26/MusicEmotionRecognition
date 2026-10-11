
# Setup Guide

This guide explains how to set up the Music Emotion Recognition project on a new Windows computer.

## 1. Prerequisites

### Install Visual Studio Code

Download: https://code.visualstudio.com/

### Install Git

Download: https://git-scm.com/downloads/win

After installation, open PowerShell and verify:

```powershell
git --version
```

### Install Python

Download: https://www.python.org/downloads/windows/

The required Python version will be specified once the project environment is finalized.

Verify the installation:

```powershell
py --version
```

## 2. Clone the Repository

Open PowerShell and navigate to the directory where you want to store the project.

For example:

```powershell
cd $HOME\Documents
```

Clone the repository:

```powershell
git clone https://github.com/dianaG26/MusicEmotionRecognition.git
```

Since this repository is private, you must authenticate with a GitHub account that has access.

Navigate into the project:

```powershell
cd MusicEmotionRecognition
```

Open it in VS Code:

```powershell
code .
```

If the `code` command is unavailable, open VS Code and select File > Open Folder.

## 3. Python Virtual Environment

Once the supported Python version has been established, create a virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install project dependencies (after requirements.txt is added):

```powershell
python -m pip install -r requirements.txt
```

## 4. Dataset

Planned dataset: MERGE Audio Balanced.

Official source:
https://zenodo.org/records/13939205

Dataset download, verification and preprocessing instructions will be added after the dataset is inspected.

## 5. Model Training

See [TRAINING.md](TRAINING.md).

## 6. Running the Application

Backend and frontend instructions will be added as the application is developed.
