from typing import Literal
import dearpygui.dearpygui as dpg
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import tyro
from dataclasses import dataclass


@dataclass
class FamudyViewerConfig:
    root_folder: Path
    mode: Literal['multi-frame', 'single-frame']='multi-frame'
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

        if self.mode == 'multi-frame':
            self.filter_func = lambda x: x == 'f'
        elif self.mode == 'single-frame':
            self.filter_func = lambda x: x == 'o'
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # load csv
        csv_path = self.root_folder / 'processing_status_flame.csv'
        self.data = pd.read_csv(csv_path, sep='\t', header=1)

        self.data.iloc[:, 0] = self.data.iloc[:, 0].map(lambda x: f'{x:03d}')
        self.data.set_index("ID", inplace=True)

        # init variables
        self.subjects = self.data.index.tolist()
        self.subjects = [subject for subject in self.subjects if len(list(filter(self.filter_func, self.data.loc[subject].values))) > 0]

        self.sequences = self.data.columns[1:].tolist()
        self.sequences = [sequence for sequence in self.sequences if len(list(filter(self.filter_func, self.data.loc[:, sequence].values))) > 0]
        
        self.selected_subject = '-'
        self.selected_sequence = '-'

        self.available_subjects = self.subjects
        self.available_sequences = self.sequences

    def reset_folder_tree(self, update_items=True):
        self.timesteps = []
        self.selected_timestep = ''
        self.selected_timestep_idx = 0
        self.filetypes = []
        self.selected_filetype = ''
        self.cameras = []
        self.selected_camera = ''
        if update_items:
            # dpg.configure_item("combo_timestep", items=self.timesteps, default_value=self.selected_timestep)
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
                    # dpg.configure_item("listbox_sequence", items=['-'] + self.available_sequences, default_value=self.selected_sequence)
                    self.update_folder_tree(level='timestep')
                    self.need_update = True
                dpg.add_combo(['-'] + self.subjects, default_value=self.selected_subject, label="subject  ", height_mode=dpg.mvComboHeight_Large, callback=set_subject, tag='combo_subject')
                # dpg.add_listbox(['-'] + self.subjects, label="subject", default_value=self.selected_subject, num_items=5, callback=set_subject, tag='listbox_subject', tracked=True)
                
                def prev_subject(sender, data):
                    if self.selected_subject == '-':
                        self.selected_subject = self.available_subjects[-1]
                    else:
                        idx = self.available_subjects.index(self.selected_subject)
                        if idx > 0:
                            self.selected_subject = self.available_subjects[idx-1]
                        else:
                            self.selected_subject = self.available_subjects[-1]
                    dpg.set_value("combo_subject", value=self.selected_subject)
                    # dpg.set_value("listbox_subject", value=self.selected_subject)
                    set_subject(None, self.selected_subject)
                dpg.add_button(label="W", callback=prev_subject, width=19)

                def next_subject(sender, data):
                    if self.selected_subject == '-':
                        self.selected_subject = self.available_subjects[0]
                    else:
                        idx = self.available_subjects.index(self.selected_subject)
                        if idx < len(self.available_subjects)-1:
                            self.selected_subject = self.available_subjects[idx+1]
                        else:
                            self.selected_subject = self.available_subjects[0]
                    dpg.set_value("combo_subject", value=self.selected_subject)
                    # dpg.set_value("listbox_subject", value=self.selected_subject)
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
                    # dpg.configure_item("listbox_subject", items=['-'] + self.available_subjects, default_value=self.selected_subject)
                    self.update_folder_tree(level='timestep')
                    self.need_update = True
                dpg.add_combo(['-'] + self.sequences, default_value=self.selected_sequence, label="sequence ", height_mode=dpg.mvComboHeight_Large, callback=set_sequence, tag='combo_sequence')
                # dpg.add_listbox(['-'] + self.sequences, label="sequence", default_value=self.selected_sequence, num_items=5, callback=set_sequence, tag='listbox_sequence', tracked=True)

                def prev_sequence(sender, data):
                    if self.selected_sequence == '-':
                        self.selected_sequence = self.available_sequences[-1]
                    else:
                        idx = self.available_sequences.index(self.selected_sequence)
                        if idx > 0:
                            self.selected_sequence = self.available_sequences[idx-1]
                        else:
                            self.selected_sequence = self.available_sequences[-1]
                    dpg.set_value("combo_sequence", value=self.selected_sequence)
                    # dpg.set_value("listbox_sequence", value=self.selected_sequence)
                    set_sequence(None, self.selected_sequence)
                dpg.add_button(label="A", callback=prev_sequence, width=19)
                
                def next_sequence(sender, data):
                    if self.selected_sequence == '-':
                        self.selected_sequence = self.available_sequences[0]
                    else:
                        idx = self.available_sequences.index(self.selected_sequence)
                        if idx < len(self.available_sequences)-1:
                            self.selected_sequence = self.available_sequences[idx+1]
                        else:
                            self.selected_sequence = self.available_sequences[0]
                    dpg.set_value("combo_sequence", value=self.selected_sequence)
                    # dpg.set_value("listbox_sequence", value=self.selected_sequence)
                    set_sequence(None, self.selected_sequence)
                dpg.add_button(label="D", callback=next_sequence, width=19)


            # timestep switch
            with dpg.group(horizontal=True):
                # def set_timestep(sender, data):
                #     self.selected_timestep = data
                #     self.update_folder_tree(level='filetype')
                    # self.need_update = True
                # dpg.add_combo([], label="time step", height_mode=dpg.mvComboHeight_Large, callback=set_timestep, tag='combo_timestep')

                def set_timestep_slider(sender, data):
                    self.selected_timestep_idx = data
                    self.selected_timestep = self.timesteps[self.selected_timestep_idx]
                    self.update_folder_tree(level='filetype')
                    self.need_update = True
                dpg.add_slider_int(label="time step", max_value=len(self.timesteps), callback=set_timestep_slider, tag='slider_timestep')

                def prev_timestep(sender, data):
                    if dpg.is_item_focused("text_path"):
                        return
                    if len(self.timesteps) > 1:
                        if self.selected_timestep_idx > 0:
                            self.selected_timestep_idx -= 1
                        else:
                            self.selected_timestep_idx = len(self.timesteps)-1
                        self.selected_timestep = self.timesteps[self.selected_timestep_idx]
                        dpg.set_value("slider_timestep", value=self.selected_timestep_idx)
                        set_timestep_slider(None, self.selected_timestep_idx)
                dpg.add_button(label="Button", callback=prev_timestep, arrow=True, direction=dpg.mvDir_Left)

                def next_timestep(sender, data):
                    if dpg.is_item_focused("text_path"):
                        return
                    print(self.timesteps)
                    if len(self.timesteps) > 1:
                        if self.selected_timestep_idx < len(self.timesteps)-1:
                            self.selected_timestep_idx += 1
                        else:
                            self.selected_timestep_idx = 0
                        self.selected_timestep = self.timesteps[self.selected_timestep_idx]
                        dpg.set_value("slider_timestep", value=self.selected_timestep_idx)
                        set_timestep_slider(None, self.selected_timestep_idx)            
                dpg.add_button(label="Button", callback=next_timestep, arrow=True, direction=dpg.mvDir_Right)


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
                    self.need_update = True
                dpg.add_combo([], label="camera   ", height_mode=dpg.mvComboHeight_Large, callback=set_camera, tag='combo_camera')

                def prev_camera(sender, data):
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
                    if len(self.cameras) > 0:
                        idx = self.cameras.index(self.selected_camera)
                        if idx < len(self.cameras)-1:
                            self.selected_camera = self.cameras[idx+1]
                        else:
                            self.selected_camera = self.cameras[0]
                        dpg.set_value("combo_camera", value=self.selected_camera)
                        set_camera(None, self.selected_camera)
                dpg.add_button(label="Button", callback=next_camera, arrow=True, direction=dpg.mvDir_Down)

            # path
            dpg.add_input_text(label="path", default_value='', tag='text_path', readonly=True)
        
        # key press handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_W, callback=prev_subject)
            dpg.add_key_press_handler(dpg.mvKey_S, callback=next_subject)
            dpg.add_key_press_handler(dpg.mvKey_A, callback=next_sequence)
            dpg.add_key_press_handler(dpg.mvKey_D, callback=prev_sequence)
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=prev_timestep)
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=next_timestep)
            dpg.add_key_press_handler(dpg.mvKey_Up, callback=prev_camera)
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=next_camera)

        # theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("viewer_tag", theme_no_padding)

        dpg.create_viewport(title='Famudy Viewer', width=self.width, height=self.height, resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

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
    
    def update_folder_tree(self, level=Literal['timestep', 'filetype', 'camera']):
        if self.selected_sequence == '-' or self.selected_subject == '-':
            self.reset_folder_tree()
            return
        
        update_time_step = level in ['timestep']
        update_filetype = level in ['timestep', 'filetype']
        update_camera = level in ['timestep', 'filetype', 'camera']

        if update_time_step:
            self.timesteps = self.iterdir(self.selected_subject, self.selected_sequence)
            self.selected_timestep = self.timesteps[0]
            self.selected_timestep_idx = 0
            # dpg.configure_item("combo_timestep", items=self.timesteps, default_value=self.selected_timestep)
            dpg.configure_item("slider_timestep", max_value=len(self.timesteps)-1, default_value=self.selected_timestep_idx)
            

        if update_filetype:
            self.filetypes = self.iterdir(self.selected_subject, self.selected_sequence, self.selected_timestep)
            self.filetypes_images = sorted([x for x in self.filetypes if 'image' in x], reverse=True)
            self.selected_filetype = self.filetypes_images[0] if len(self.filetypes_images) > 0 else self.filetypes[0]
            dpg.configure_item("combo_filetype", items=self.filetypes, default_value=self.selected_filetype)

        if update_camera:
            self.cameras = self.iterdir(self.selected_subject, self.selected_sequence, self.selected_timestep, self.selected_filetype)
            self.selected_camera = self.cameras[0]
            dpg.configure_item("combo_camera", items=self.cameras, default_value=self.selected_camera)

    def update_viewer(self, refresh=False):
        if self.selected_sequence != '-' and self.selected_subject != '-':
            path = self.root_folder / self.selected_subject / 'sequences' / self.selected_sequence / 'timesteps' / self.selected_timestep / self.selected_filetype / self.selected_camera
            img = self.load_image(path)
            dpg.set_value("texture_tag", img)
        else:
            img = np.zeros([self.height, self.width, 3])
            dpg.set_value("texture_tag", img)

    def load_image(self, path):
        dpg.set_value("text_path", path)
        img = Image.open(path)
        scale = min(self.height / img.height, self.width / img.width)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = np.pad(img, ((0, self.height - img.shape[0]), (0, self.width - img.shape[1]), (0, 0)), mode='constant', constant_values=0)
        return img.astype(np.float32) / 255


if __name__ == '__main__':
    cfg = tyro.cli(FamudyViewerConfig)
    app = FamudyViewer(cfg)
    app.run()
