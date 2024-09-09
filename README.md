# NeRSemble Data Viewer

A GUI program to inspect images in the [NeRSemble](https://tobias-kirschstein.github.io/nersemble/) dataset.

<div align="center">
        <image src="./screenshot.png" height=800px></image>
</div>

## Installation

```shell
pip install -r requirements.txt
```

## Usage

```shell
python main.py --root_folder <root_folder>
```

## Expected folder hierarchy

```
<root_folder>
|- 017
        |- EXP-1-head
                |- images
                        |- cam_220700191_000000.jpg
                        |- cam_220700191_000001.jpg
                        |- ...
```
