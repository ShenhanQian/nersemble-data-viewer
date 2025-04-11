# NeRSemble Data Viewer

A GUI program to inspect images in the [NeRSemble](https://tobias-kirschstein.github.io/nersemble/) dataset.

> [!NOTE]
> This program has been adapted to NeRSemble Dataset V2. The folder structure is no longer compatible with NeRSemble Dataset V1.

<div align="center">
        <image src="./screenshot.png" height=800px></image>
</div>

## Installation

```shell
pip install -r requirements.txt
```

## Usage

Please first extract frames from the raw videos with [this script](https://github.com/ShenhanQian/VHAP/blob/main/doc/nersemble_v2.md#1-preprocess). Then you can view them with

```shell
python main.py --root_folder <root_folder>
```

## Expected folder hierarchy

```
<root_folder>
|- 017
        |- sequences
                |- EXP-1-head
                        |- images
                                |- cam_220700191_000000.jpg
                                |- cam_220700191_000001.jpg
                                |- ...
```
