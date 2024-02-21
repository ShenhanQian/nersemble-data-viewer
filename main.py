from typing import Literal, Optional
import dearpygui.dearpygui as dpg
from pathlib import Path
import glob
import numpy as np
from PIL import Image
import pandas as pd
import tyro
from dataclasses import dataclass
import cv2
import matplotlib.pyplot as plt


region2seg_class = {
    'bg': 0,
    'skin': 1,
    'l_brow': 2,
    'r_brow': 3,
    'l_eye': 4,
    'r_eye': 5,
    'eye_g': 6,
    'l_ear': 7,
    'r_ear': 8,
    'ear_r': 9,
    'nose': 10,
    'mouth': 11,
    'u_lip': 12,
    'l_lip': 13,
    'neck': 14,
    'neck_l': 15,
    'cloth': 16,
    'hair': 17,
    'hat': 18
}


@dataclass
class FamudyViewerConfig:
    root_folder: Path
    mode: str='XoS'
    width: int=1000
    height: int=1000

class FamudyViewer(object):
    def __init__(self, cfg):
        self.root_folder = cfg.root_folder
        self.mode = cfg.mode
        self.width = cfg.width
        self.height = cfg.height

        self.width_nav = 390
        self.height_nav = 400
        self.need_update = False

        self.filter_func = lambda x: x in self.mode
        
        # load csv
        csv_path = self.root_folder / 'processing_status.csv'
        self.data = pd.read_csv(csv_path, sep='\t', header=1)

        self.data.iloc[:, 0] = self.data.iloc[:, 0].map(lambda x: f'{x:03d}')
        self.data.set_index("ID", inplace=True)

        # init variables
        self.subjects = self.data.index.tolist()
        self.subjects = [subject for subject in self.subjects if len(list(filter(self.filter_func, self.data.loc[subject].values))) > 0]

        self.sequences = self.data.columns.tolist()
        self.sequences = [sequence for sequence in self.sequences if len(list(filter(self.filter_func, self.data.loc[:, sequence].values))) > 0]

        self.reset_subject_sequence(update_items=False)

    def reset_subject_sequence(self, update_items=True):
        self.selected_subject = '-'
        self.selected_sequence = '-'
        self.available_subjects = self.subjects
        self.available_sequences = self.sequences

        if update_items:
            dpg.configure_item("combo_subject", items=['-'] + self.available_subjects, default_value=self.selected_subject)
            dpg.configure_item("combo_sequence", items=['-'] + self.available_sequences, default_value=self.selected_sequence)

    def reset_folder_tree(self, update_items=True):
        self.timesteps = []
        self.selected_timestep = ''
        self.selected_timestep_idx = 0
        self.filetypes = []
        self.selected_filetype = ''
        self.cameras = []
        self.selected_camera = ''
        if update_items:
            dpg.configure_item("slider_timestep", default_value=self.selected_timestep_idx, max_value=0, min_value=0)
            dpg.configure_item("combo_filetype", items=self.filetypes, default_value=self.selected_filetype)
            dpg.configure_item("combo_camera", items=self.cameras, default_value=self.selected_camera)

    def define_gui(self):
        dpg.create_context()

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self.width, height=self.height, default_value=np.zeros([self.height, self.width, 3]), format=dpg.mvFormat_Float_rgb, tag="texture_tag")

        # viewer window
        with dpg.window(label="Viewer", pos=[0, 0], tag='viewer_tag', width=self.width, height=self.height, no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True):
            dpg.add_image("texture_tag", tag='image_tag', width=self.width, height=self.height)

        # navigator window
        with dpg.window(label="Navigator", tag='navigator_tag', width=self.width_nav, pos=[self.width-self.width_nav-15, 0], autosize=True):

            # subject switch
            with dpg.group(horizontal=True):
                def set_subject(sender, data):
                    self.selected_subject = data
                    if data == '-':
                        self.available_sequences = self.sequences
                    else:
                        self.available_sequences = [sequence for sequence, v in self.data.loc[data].items() if self.filter_func(v)]
                        
                        if self.selected_sequence not in self.available_sequences:
                            if len(self.available_sequences) > 0:
                                self.selected_sequence = self.available_sequences[0]
                            else:
                                self.selected_sequence == '-'

                    dpg.configure_item("combo_sequence", items=['-'] + self.available_sequences, default_value=self.selected_sequence)
                    self.check_calibration()
                    self.update_folder_tree(level='timestep')
                    self.need_update = True
                dpg.add_combo(['-'] + self.subjects, default_value=self.selected_subject, label="subject  ", height_mode=dpg.mvComboHeight_Large, callback=set_subject, tag='combo_subject')
                
                def prev_subject(sender, data):
                    if dpg.is_item_focused("text_mode"):
                        return
                    if self.selected_subject == '-':
                        self.selected_subject = self.available_subjects[-1]
                    else:
                        idx = self.available_subjects.index(self.selected_subject)
                        if idx > 0:
                            self.selected_subject = self.available_subjects[idx-1]
                        else:
                            self.selected_subject = self.available_subjects[-1]
                    dpg.set_value("combo_subject", value=self.selected_subject)
                    set_subject(None, self.selected_subject)
                dpg.add_button(label="W", callback=prev_subject, width=19)

                def next_subject(sender, data):
                    if dpg.is_item_focused("text_mode"):
                        return
                    if self.selected_subject == '-':
                        self.selected_subject = self.available_subjects[0]
                    else:
                        idx = self.available_subjects.index(self.selected_subject)
                        if idx < len(self.available_subjects)-1:
                            self.selected_subject = self.available_subjects[idx+1]
                        else:
                            self.selected_subject = self.available_subjects[0]
                    dpg.set_value("combo_subject", value=self.selected_subject)
                    set_subject(None, self.selected_subject)
                dpg.add_button(label="S", callback=next_subject, width=19)
            
            # sequence switch
            with dpg.group(horizontal=True):
                def set_sequence(sender, data):
                    self.selected_sequence = data
                    if data == '-':
                        self.available_subjects = self.subjects
                    else:
                        self.available_subjects = [subject for subject, v in self.data.loc[:, data].items() if self.filter_func(v)]

                        if self.selected_subject not in self.available_subjects:
                            if len(self.available_subjects) > 0:
                                self.selected_subject = self.available_subjects[0]
                            else:
                                self.selected_subject == '-'
                    
                    dpg.configure_item("combo_subject", items=['-'] + self.available_subjects, default_value=self.selected_subject)
                    self.update_folder_tree(level='timestep')
                    self.need_update = True
                dpg.add_combo(['-'] + self.sequences, default_value=self.selected_sequence, label="sequence ", height_mode=dpg.mvComboHeight_Large, callback=set_sequence, tag='combo_sequence')

                def prev_sequence(sender, data):
                    if dpg.is_item_focused("text_mode"):
                        return
                    if self.selected_sequence == '-':
                        self.selected_sequence = self.available_sequences[-1]
                    else:
                        idx = self.available_sequences.index(self.selected_sequence)
                        if idx > 0:
                            self.selected_sequence = self.available_sequences[idx-1]
                        else:
                            self.selected_sequence = self.available_sequences[-1]
                    dpg.set_value("combo_sequence", value=self.selected_sequence)
                    set_sequence(None, self.selected_sequence)
                dpg.add_button(label="A", callback=prev_sequence, width=19)
                
                def next_sequence(sender, data):
                    if dpg.is_item_focused("text_mode"):
                        return
                    if self.selected_sequence == '-':
                        self.selected_sequence = self.available_sequences[0]
                    else:
                        idx = self.available_sequences.index(self.selected_sequence)
                        if idx < len(self.available_sequences)-1:
                            self.selected_sequence = self.available_sequences[idx+1]
                        else:
                            self.selected_sequence = self.available_sequences[0]
                    dpg.set_value("combo_sequence", value=self.selected_sequence)
                    set_sequence(None, self.selected_sequence)
                dpg.add_button(label="D", callback=next_sequence, width=19)


            # timestep switch
            with dpg.group(horizontal=True):
                def set_timestep_slider(sender, data):
                    self.selected_timestep_idx = data
                    self.selected_timestep = self.timesteps[self.selected_timestep_idx]
                    self.update_folder_tree(level='filetype')
                    self.need_update = True
                dpg.add_slider_int(label="time step", max_value=len(self.timesteps), callback=set_timestep_slider, tag='slider_timestep')

                def prev_timestep(sender, data):
                    if dpg.is_item_focused("text_mode"):
                        return
                    if len(self.timesteps) > 1:
                        if self.selected_timestep_idx > 0:
                            self.selected_timestep_idx -= 1
                        self.selected_timestep = self.timesteps[self.selected_timestep_idx]
                        dpg.set_value("slider_timestep", value=self.selected_timestep_idx)
                        set_timestep_slider(None, self.selected_timestep_idx)
                dpg.add_button(label="Button", callback=prev_timestep, arrow=True, direction=dpg.mvDir_Left)

                def next_timestep(sender, data):
                    if dpg.is_item_focused("text_mode"):
                        return
                    if len(self.timesteps) > 1:
                        if self.selected_timestep_idx < len(self.timesteps)-1:
                            self.selected_timestep_idx += 1
                        self.selected_timestep = self.timesteps[self.selected_timestep_idx]
                        dpg.set_value("slider_timestep", value=self.selected_timestep_idx)
                        set_timestep_slider(None, self.selected_timestep_idx)            
                dpg.add_button(label="Button", callback=next_timestep, arrow=True, direction=dpg.mvDir_Right)

                def set_timestep(sender, data):
                    if self.selected_subject == '-' or self.selected_sequence == '-':
                        return
                    if dpg.is_item_focused("text_mode"):
                        return
                    
                    if sender == 'mvKey_Home':
                        self.selected_timestep_idx = 0
                    elif sender == 'mvKey_End':
                        if len(self.timesteps) > 0:
                            self.selected_timestep_idx = len(self.timesteps)-1
                    self.selected_timestep = self.timesteps[self.selected_timestep_idx]
                    dpg.set_value("slider_timestep", value=self.selected_timestep_idx)
                    set_timestep_slider(None, self.selected_timestep_idx)       


            # filetype switch
            def set_filetype(sender, data):
                self.selected_filetype = data
                self.update_folder_tree(level='camera')
                self.need_update = True
            dpg.add_combo([], label="file type", height_mode=dpg.mvComboHeight_Large, callback=set_filetype, tag='combo_filetype')


            # camera switch
            with dpg.group(horizontal=True):
                def set_camera(sender, data):
                    self.selected_camera = data
                    self.update_folder_tree(level='camera')
                    self.need_update = True
                dpg.add_combo([], label="camera   ", height_mode=dpg.mvComboHeight_Large, callback=set_camera, tag='combo_camera')

                def prev_camera(sender, data):
                    if dpg.is_item_focused("text_mode"):
                        return
                    if len(self.cameras) > 0:
                        idx = self.cameras.index(self.selected_camera)
                        if idx > 0:
                            self.selected_camera = self.cameras[idx-1]
                        else:
                            self.selected_camera = self.cameras[-1]
                        dpg.set_value("combo_camera", value=self.selected_camera)
                        set_camera(None, self.selected_camera)
                dpg.add_button(label="", callback=prev_camera, arrow=True, direction=dpg.mvDir_Up)

                def next_camera(sender, data):
                    if dpg.is_item_focused("text_mode"):
                        return
                    if len(self.cameras) > 0:
                        idx = self.cameras.index(self.selected_camera)
                        if idx < len(self.cameras)-1:
                            self.selected_camera = self.cameras[idx+1]
                        else:
                            self.selected_camera = self.cameras[0]
                        dpg.set_value("combo_camera", value=self.selected_camera)
                        set_camera(None, self.selected_camera)
                dpg.add_button(label="Button", callback=next_camera, arrow=True, direction=dpg.mvDir_Down)

            # filter mode
            def set_mode(sender, data):
                self.mode = data
                self.reset_subject_sequence()
                self.update_folder_tree()
                self.update_viewer()
            dpg.add_input_text(label="filter mode", default_value=self.mode, tag='text_mode', callback=set_mode)
            dpg.add_text("o: all timesteps (not processed)\n"
                         "x: all timesteps (processed)\n"
                         "S: all timesteps (only the first frame processed)\n"
                         "s: single timestep\n"
                         "f: no FLAME param"
                        )
            
            dpg.add_text("", tag='text_calibration', color=[255, 0, 0])

            # annotations
            dpg.add_separator()
            def set_annotation(sender, data):
                self.need_update = True
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="landmarks (STAR)  ", callback=set_annotation, tag='checkbox_lmk_star', show=False, default_value=False)
                dpg.add_checkbox(label="landmarks (PIPnet)", callback=set_annotation, tag='checkbox_lmk_pipnet', show=False, default_value=False)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="landmarks (FA)    ", callback=set_annotation, tag='checkbox_lmk_fa', show=False, default_value=False)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="foreground        ", callback=set_annotation, tag='checkbox_fg', show=False, default_value=False)
                dpg.add_checkbox(label="segmentation      ", callback=set_annotation, tag='checkbox_seg', show=False, default_value=False)
            
            with dpg.collapsing_header(label="Filter regions", default_open=False, show=False, tag='collapsing_filter_regions'):
                def set_region(sender, data):
                    self.need_update = True

                n_cols = 5
                n_rows = len(region2seg_class) // n_cols + int((len(region2seg_class) % n_cols) > 0)
                
                for i in range(n_rows):
                    dpg.add_group(tag=f'filter_region_group_{i}', horizontal=True)

                for i, region in enumerate(region2seg_class.keys()):
                    dpg.add_checkbox(label=f'{region:6s}', callback=set_region, tag=f'checkbox_{region}', show=True, default_value=True, parent=f'filter_region_group_{i // n_cols}')


        
        # key press handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_W, callback=prev_subject)
            dpg.add_key_press_handler(dpg.mvKey_S, callback=next_subject)
            dpg.add_key_press_handler(dpg.mvKey_A, callback=prev_sequence)
            dpg.add_key_press_handler(dpg.mvKey_D, callback=next_sequence)
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=prev_timestep)
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=next_timestep)
            dpg.add_key_press_handler(dpg.mvKey_Up, callback=prev_camera)
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=next_camera)
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=set_timestep, tag='mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=set_timestep, tag='mvKey_End')

        # theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("viewer_tag", theme_no_padding)

    def resize_windows(self):
        dpg.configure_item('viewer_tag', width=self.width, height=self.height)
        dpg.configure_item('navigator_tag', width=self.width_nav, height=self.height_nav, pos=[self.width-self.width_nav-15, 0])

        dpg.delete_item('texture_tag')
        dpg.delete_item('image_tag')
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self.width, height=self.height, default_value=np.zeros([self.height, self.width, 3]), format=dpg.mvFormat_Float_rgb, tag="texture_tag")
        dpg.add_image("texture_tag", tag='image_tag', parent='viewer_tag')
        self.need_update = True

    def run(self):
        self.reset_folder_tree(update_items=False)
        self.define_gui()
        dpg.create_viewport(title='Famudy Viewer', width=self.width, height=self.height, resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            if self.width != dpg.get_viewport_width() or self.height != dpg.get_viewport_height():
                self.width = dpg.get_viewport_width()
                self.height = dpg.get_viewport_height()
                self.resize_windows()

            if self.need_update:
                self.update_viewer()
                self.need_update = False
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
    
    def iterdir(self, subject=None, sequence=None, timestep=None, filetype=None):
        if filetype is not None:
            return [x.name for x in (self.root_folder / subject / 'sequences' / sequence / 'timesteps' / timestep / filetype).iterdir()]
        elif timestep is not None:
            return [x.name for x in (self.root_folder / subject / 'sequences' / sequence / 'timesteps' / timestep).iterdir()]
        elif sequence is not None:
            return [x.name for x in (self.root_folder / subject / 'sequences' / sequence / 'timesteps').iterdir()]
        elif subject is not None:
            return [x.name for x in (self.root_folder / subject / 'sequences').iterdir()]
        else:
            raise ValueError("Invalid arguments")
    
    def check_calibration(self):
        no_calibration = False
        if self.selected_subject != '-':
            calibration_path = self.root_folder / self.selected_subject / 'calibration' / 'calibration_result.json'
            if not calibration_path.exists():
                no_calibration = True
        dpg.set_value("text_calibration", value="no calibration" if no_calibration else "")
    
    def update_annotations(self):
        lmk_star_path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'annotations' / 'landmarks2D' / 'STAR' / f"{self.selected_camera}.npz"
        if lmk_star_path.exists():
            dpg.configure_item('checkbox_lmk_star', show=True)
        else:
            dpg.configure_item('checkbox_lmk_star', show=False)

        lmk_pipnet_path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'annotations' / 'landmarks2D' / 'PIPnet' / f"{self.selected_camera.replace('cam_', '')}.npy"
        if lmk_pipnet_path.exists():
            dpg.configure_item('checkbox_lmk_pipnet', show=True)
        else:
            dpg.configure_item('checkbox_lmk_pipnet', show=False)

        lmk_fa_path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'annotations' / 'landmarks2D' / 'face-alignment' / f"{self.selected_camera}.npz"
        if lmk_fa_path.exists():
            dpg.configure_item('checkbox_lmk_fa', show=True)
        else:
            dpg.configure_item('checkbox_lmk_fa', show=False)
        
        fg = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'timesteps' / self.selected_timestep / 'alpha_map'
        if fg.exists():
            dpg.configure_item('checkbox_fg', show=True)
        else:
            dpg.configure_item('checkbox_fg', show=False)
        
        seg = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'timesteps' / self.selected_timestep / 'bisenet_segmentation_masks'
        if seg.exists():
            dpg.configure_item('checkbox_seg', show=True)
            dpg.configure_item('collapsing_filter_regions', show=True)
        else:
            dpg.configure_item('checkbox_seg', show=False)
            dpg.configure_item('collapsing_filter_regions', show=False)

    
    def update_folder_tree(self, level=Optional[Literal['timestep', 'filetype', 'camera']]):
        if self.selected_sequence == '-' or self.selected_subject == '-':
            self.reset_folder_tree()
            return
        
        update_time_step = level in ['timestep']
        update_filetype = level in ['timestep', 'filetype']
        update_camera = level in ['timestep', 'filetype', 'camera']

        if update_time_step:
            self.timesteps = sorted(self.iterdir(self.selected_subject, self.selected_sequence))
            if self.selected_timestep not in self.timesteps:
                self.selected_timestep = self.timesteps[0]
            self.selected_timestep_idx = self.timesteps.index(self.selected_timestep)
            dpg.configure_item("slider_timestep", max_value=len(self.timesteps)-1, default_value=self.selected_timestep_idx)

        if update_filetype:
            self.filetypes = self.iterdir(self.selected_subject, self.selected_sequence, self.selected_timestep)
            self.filetypes_images = sorted([x for x in self.filetypes if 'image' in x], reverse=True)
            if self.selected_filetype not in self.filetypes:
                self.selected_filetype = self.filetypes_images[0] if len(self.filetypes_images) > 0 else self.filetypes[0]
            dpg.configure_item("combo_filetype", items=self.filetypes, default_value=self.selected_filetype)

        if update_camera:
            self.cameras = [f.split('.')[0] for f in self.iterdir(self.selected_subject, self.selected_sequence, self.selected_timestep, self.selected_filetype)]
            if self.selected_camera not in self.cameras:
                self.selected_camera = self.cameras[0]
            dpg.configure_item("combo_camera", items=self.cameras, default_value=self.selected_camera)
        
        self.update_annotations()

    def update_viewer(self):
        if self.selected_sequence != '-' and self.selected_subject != '-':
            
            path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'timesteps' / self.selected_timestep / self.selected_filetype
            path = glob.glob(f'{str(path)}/*{self.selected_camera}*')
            if len(path) == 0:
                return
            
            path = path[0]
            if 'jpg' not in path.lower() and 'png' not in path.lower():
                return
            
            if self.selected_filetype == 'bisenet_segmentation_masks':
                # load as uint8 and get float32 after applying colormap
                img = self.load_image(path, Image.NEAREST)
                cm = plt.get_cmap('tab20c')
                img = cm(img[:, :, 0])[:, :, :3].astype(np.float32)
            else:
                # directly load as float32
                img = self.load_image(path)
                img = img.astype(np.float32) / 255
            
            if dpg.get_item_configuration("checkbox_lmk_star")['show'] and dpg.get_value("checkbox_lmk_star"):
                npz_path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'annotations' / 'landmarks2D' / 'STAR' / f"{self.selected_camera}.npz"
                npz = np.load(npz_path)
                lmk_star = npz['face_landmark_2d']
                bbox_star = npz['bounding_box']

                color = (0, 255, 0)
                for lmk in lmk_star[self.selected_timestep_idx]:
                    x = int(lmk[0] * img.shape[1])
                    y = int(lmk[1] * img.shape[0])
                    cv2.circle(img, (x, y), 2, color, -1)

                # x1, y1, x2, y2 = bbox_star[self.selected_timestep_idx][:4]
                # x1 = int(x1 * img.shape[1])
                # y1 = int(y1 * img.shape[0])
                # x2 = int(x2 * img.shape[1])
                # y2 = int(y2 * img.shape[0])
                # cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

            if dpg.get_item_configuration("checkbox_lmk_pipnet")['show'] and dpg.get_value("checkbox_lmk_pipnet"):
                npy_path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'annotations' / 'landmarks2D' / 'PIPnet' / f"{self.selected_camera.replace('cam_', '')}.npy"
                lmk_pipnet = np.load(npy_path)

                color = (0, 0, 255)
                for lmk in lmk_pipnet[self.selected_timestep_idx]:
                    x = int(lmk[0] * img.shape[1])
                    y = int(lmk[1] * img.shape[0])
                    cv2.circle(img, (x, y), 2, color, -1)
            
            if dpg.get_item_configuration("checkbox_lmk_fa")['show'] and dpg.get_value("checkbox_lmk_fa"):
                npz_path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'annotations' / 'landmarks2D' / 'face-alignment' / f"{self.selected_camera}.npz"
                npz = np.load(npz_path)
                lmk_fa = npz['face_landmark_2d']
                bbox_fa = npz['bounding_box']

                color = (255, 0, 0)
                for lmk in lmk_fa[self.selected_timestep_idx]:
                    x = int(lmk[0] * img.shape[1])
                    y = int(lmk[1] * img.shape[0])
                    cv2.circle(img, (x, y), 2, color, -1)

                # x1, y1, x2, y2 = bbox_fa[self.selected_timestep_idx][:4]
                # x1 = int(x1 * img.shape[1])
                # y1 = int(y1 * img.shape[0])
                # x2 = int(x2 * img.shape[1])
                # y2 = int(y2 * img.shape[0])
                # cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            
            if dpg.get_item_configuration("checkbox_fg")['show'] and dpg.get_value("checkbox_fg"):
                fg_path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'timesteps' / self.selected_timestep / 'alpha_map' / f'{self.selected_camera}.png'
                fg_alpha = self.load_image(fg_path).astype(np.float32)[..., :3] / 255
                img = img * (fg_alpha) + np.ones_like(img) * (1 - fg_alpha)
            
            if dpg.get_item_configuration("checkbox_seg")['show'] and dpg.get_value("checkbox_seg"):
                seg_path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'timesteps' / self.selected_timestep / 'bisenet_segmentation_masks' / f'segmentation_{self.selected_camera}.png'
                seg = self.load_image(seg_path, Image.NEAREST)
                cm = plt.get_cmap('tab20c')
                seg = cm(seg[:, :, 0])[:, :, :3].astype(np.float32)
                img = img * 0.5 + seg * 0.5
            
            if dpg.get_item_configuration("collapsing_filter_regions")['show']:
                seg_path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'timesteps' / self.selected_timestep / 'bisenet_segmentation_masks' / f'segmentation_{self.selected_camera}.png'
                seg = self.load_image(seg_path, Image.NEAREST)
                for region, seg_class in region2seg_class.items():
                    if not dpg.get_value(f"checkbox_{region}"):
                        mask = np.ones_like(img)
                        mask[seg == seg_class] = 0
                        img = img * mask + np.ones_like(img) * (1 - mask)
            
            img = np.pad(img, ((0, self.height - img.shape[0]), (0, self.width - img.shape[1]), (0, 0)), mode='constant', constant_values=0)

            dpg.set_value("texture_tag", img)

    def load_image(self, path, resample=Image.BILINEAR):
        img = Image.open(path)
        scale = min(self.height / img.height, self.width / img.width)
        img = img.resize((int(img.width * scale), int(img.height * scale)), resample)
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        return img


if __name__ == '__main__':
    cfg = tyro.cli(FamudyViewerConfig)
    app = FamudyViewer(cfg)
    app.run()
