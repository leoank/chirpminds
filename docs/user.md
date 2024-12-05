# Steps
## Clone the repo 
- generate the key using git bash 
```bash
ssh-keygen -o
```
- copy the key to the git webpage : setting> ssh and gpg keys
```bash
git clone [url from code] 
```
## Create local environment
- create python virtual environment
```bash
python -m venv .venv
```
- Activate the vir environment 
```bash
.\.venv\Scripts\activate
```
- install uv manager
```bash
pip install uv
```
- install project dependancies 
```bash
uv pip install -e .[dev,notes]
```
