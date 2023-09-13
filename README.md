# Famudy Viewer

A GUI program to inspect images in the Famudy dataset.

<div>
<image src="./screenshot.png" height=500px></image>
</div>

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python main.py --root_folder <root_folder>
```

## Expected folder hierarchy
```
<root_folder>
|- 017
    |- sequences
            |- EXP-1-head
                    |- timesteps
                            |- frame_00000
                                    |- images
                                            |- cam_220700191.jpg
                                            |- cam_221501007.jpg
                                            |- ...
```
