# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# pyright: reportAny=false, reportUnknownMemberType=false, reportMissingTypeStubs=false

# %%
# | default_exp download

# %% [markdown]
# # Download files from google drive

# %%
# | export

from pathlib import Path

import gdown
from tqdm import tqdm

from chirpminds.utils import parallel

# %% [markdown]
# ## Create a file map

# %%
output_dir = Path("../../scratch/raw_data/")
file_map = {
    "1JkSXu38rHoZt-_4F-mxA8fty0U-Wij4n": str(output_dir.joinpath("interlaced.MTS"))
}
# file_map = {
#     "17s_dUURlA2mt12jIFpW5U60guCT4SveW": str(output_dir.joinpath("20230114_F02A.MP4")),
#     "1qmDbnnaIOuem_SlIdiAFF_FkeloH8ywC": str(output_dir.joinpath("20230203_F16A.MP4")),
#     "1mU4aWhU5dSYLtUUsiN0MYPy0DMzebmx_": str(output_dir.joinpath("202301211_F11A.MP4")),
#     "1wtBxT0PqsJrcsIsZknDQRtkf3VN6LpR2": str(output_dir.joinpath("20230215_F04A.MP4")),
#     "1be5q8pP7j0IpmHwZPeIq7M82ieyf8yoY": str(output_dir.joinpath("20231002_F12A.MTS")),
#     "1m8zcJUPm0KANh9wHmp-leAZtdk9fMQ4Q": str(output_dir.joinpath("20231004_F02A.MTS")),
#     "10pEQIVD5OT8rquqA1p8Az0spt-tlvaSC": str(output_dir.joinpath("MVI0012.MP4")),
#     "1wn5Oe5Qbx7RVaV0VOOkbzPHhaTdKnuM2": str(output_dir.joinpath("00000.MTS")),
# }


# %%
# | export


def download_file(
    file_name_list: list[str], file_map: dict[str, str], job_id: int
) -> None:
    for file_name in tqdm(file_name_list, position=job_id):
        gdown.download(id=file_name, output=file_map[file_name])


# %%
parallel(list(file_map.keys()), download_file, [file_map])

# %% [markdown]
# ## Check downloaded files

# %%
file_list = [file for file in output_dir.glob("*")]
print(file_list)

# %%
# | hide
import nbdev  # noqa

nbdev.nbdev_export()

# %%
