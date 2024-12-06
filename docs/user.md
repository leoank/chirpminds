# Steps

## Clone the repo

generate the key using git bash

```bash
ssh-keygen -o
```
copy the key to the git webpage : setting> ssh and gpg keys

```bash
git clone [url from code]
```
## Create local environment

create python virtual environment

```bash
python -m venv .venv
```
Activate the vir environment

```bash
.\.venv\Scripts\activate
```
install uv manager

```bash
pip install uv
```
install project dependancies

```bash
uv pip install -e .[dev,notes]
```
## Install FFMPEG system dependency

Install chocolatey package manager

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Install ffmpeg with chocolatey

```bash
choco install ffmpeg-full
```
## Sample frames from experiment video for generating segmentation dataset

Go to the scripts directory

```bash
cd chirpminds/chirpminds/scripts
```

Execute the extract frames scripts

```bash
./extract_frames.sh -s [path to input file] [path to output folder]
```

# Run automatic segmentation pipeline to generate first draft of segmentations

Go to the script directory

```bash
./label_images.py [path to input images folder] [ path to output folder]
```
cb6fd0c37537b04e945e969babc98b50b2fb1386

```powershell
$env:LOCAL_FILES_SERVING_ENABLED = "true"
```

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "C:\Users\Gayen\source\repos\chirpminds"
```
