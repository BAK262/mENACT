# Author: Ming Li <liming16@tsinghua.org.cn>
# Date: 2023/12/25
# Requirements: python=3.8, psychoPy, moviepy

from datetime import datetime
from turtle import fillcolor
from typing import List, Union, Tuple
import math
import random
import serial
import os
import pandas as pd
import numpy as np
from itertools import combinations
from psychopy import prefs
prefs.hardware['audioLib'] = ['ptb']
from psychopy.visual.movie import MovieStim
from psychopy.sound import Sound, Microphone
from psychopy.hardware.camera import Camera
from psychopy.visual.shape import ShapeStim
from psychopy.visual.rect import Rect
from psychopy.visual.slider import Slider
from psychopy.visual import Window, TextBox2, TextStim
from psychopy import core, event


class Experiment(object):
    """Base class for all experiments. New experiments should also subclass this class.

    Parameters
    ------
    exp_dir : str
        The folder path that contains all nessecary files supporting this experiment.
    exp_id : str
        Used for distinguishing files derived from different experiments.
    stimuli_screen, observe_screen : int
        Indeces of the screens for showing stimuli, and for you to observe computer activities.
    port : serial.Serial or None, default=None
        A port instance for sending trigger to the fNIRS device.
    microphone : str or None, default=None
        Name of the microphne for recording audios.
    camera : str or None, default=None
        Name of the camera for recording videos.
    camFrameSize : tuple or None, default=None
        Frame size of the recorded video.
    camFrameRate : int or None, default=None
        rame rate of the recorded video.

    Attributes
    ------
    math_guide : str
        Instruction for equation judgment task.
    math_log : pd.DataFrame
        Columns include `blockIndex`, `responseTime`, and `ifCorrect`.
    trial_log : pd.DataFrame
        Used to store information about each trial, such as completion, rating, etc. Should be overridden by subclasses that inherit this class.
    block_idx, trial_idx : int
        Changeable index of the current block and the current trial.

    Methods
    ------
    start()
        Create and open an instance of `Window` for presenting experiment stimuli.
    send_trigger()
        Send trigger(s) via `self._port`, or do nothing if `self._port` is None.
    show_instruction(text, wait, pos=(0.,0.)) -> List[str] or None
        Show instruction text in `self._win` and wait for a period.
    show_fixation(wait=5)
        Show cross-fixation in the center of `self._win` and wait for a period.
    do_math(n=10, maxTime=5.0, log=True)
        Let the subject do the equation judgment task.
    """

    def __init__(self, exp_dir: str, exp_id: str,
                 stimuli_screen: int,
                 observe_screen: int,
                 port: Union[serial.Serial, None] = None,
                 microphone: Union[str, None] = None,
                 camera: Union[str, None] = None,
                 camFrameSize: Union[Tuple[int], None] = None,
                 camFrameRate: Union[int, None] = None) -> None:
        self._exp_dir = exp_dir
        self._rawdata_dir = os.path.join(
            os.path.dirname(exp_dir), 'data', 'all_raw')
        self._exp_id = exp_id
        self._target_emotions = {'tenderness': '温情',
                                 'joy': '高兴',
                                 'inspiration': '激励',
                                 'amusement': '搞笑',
                                 'neutral': '中性',
                                 'sadness': '悲伤',
                                 'fear': '恐惧',
                                 'disgust': '恶心',
                                 'anger': '愤怒'}
        self.block_idx = 0  # Used for naming, so start with 1
        self.trial_idx = -1  # Used for indexing, so starts with 0
        self.trial_log = pd.DataFrame()

        # Set the screens
        self._stimuli_screen = stimuli_screen
        self._observe_screen = observe_screen

        # Set the external devices
        self._port = port
        if self._port:
            assert isinstance(self._port, serial.Serial)
        self._camera = camera
        self._microphone = microphone
        self._camFrameSize = camFrameSize
        self._camFrameRate = camFrameRate

        # Set the keys for quiting the experiment
        event.globalKeys.clear()
        event.globalKeys.add(key='q', modifiers=['ctrl', 'alt'],
                             func=os._exit, func_args=[1], func_kwargs=None)

    def _set_subject_info(self) -> None:
        """Record the subject's information and check its consistency if there are already some files in this `id`'s data folder."""
        from psychopy.gui import DlgFromDict
        self._subject = {'id': '',
                         'age': '',
                         'gender': ['男',
                                    '女'],
                         'hand': ['右撇子',
                                  '左撇子'],
                         'group': ['普通组',
                                   '业余组',
                                   '专业组']}
        _ = DlgFromDict(dictionary=self._subject,
                        title='请输入基本信息',
                        order=['id', 'age', 'gender', 'hand', 'group'],
                        labels={'id': '编号', 'age': '年龄', 'gender': '性别', 'hand': '利手', 'group': '组别'})
        self._subject['gender'] = 'male' if self._subject['gender'] == '男' else 'female'
        self._subject['hand'] = 'right' if self._subject['hand'] == '右撇子' else 'left'
        self._subject['group'] = 'general' if self._subject['group'] == '普通组' else 'amateur' if self._subject['group'] == '业余组' else 'professional'
        # Check the consistency of subject information
        self._save_dir = os.path.join(self._rawdata_dir, self._subject['id'])
        subjectFile = os.path.join(self._save_dir, 'subject_info.txt')
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        if not os.path.exists(subjectFile):
            with open(subjectFile, 'w') as f:
                for key, value in self._subject.items():
                    f.write(key + ':' + value + '\n')
        else:
            consistency = True
            with open(subjectFile, 'r') as f:
                for line in f.readlines():
                    key, value = line.strip('\n').split(':')
                    if self._subject[key] != value:
                        consistency = False
            assert consistency, 'Detected inconsistency of this subject''s information. Please check the input and the local file.'
        self._subject['id'] = int(self._subject['id'])
        self._subject['age'] = int(self._subject['age'])

    def _set_experiment_info(self) -> None:
        """Record the date and time when starting this experiment."""
        self._date = datetime.today().strftime('%Y%m%d')
        self._start_time = datetime.today().strftime('%H%M%S')
        self._ratingFile = os.path.join(
            self._save_dir, f'{self._exp_id}_{self._date}{self._start_time}_rating.csv')
        self._mathFile = os.path.join(
            self._save_dir, f'{self._exp_id}_{self._date}{self._start_time}_math.csv')

    def _set_math_keys(self) -> None:
        """Set the key for `equation true` and `equation false` responses during equation judgment task."""
        if (self._subject['id'] % 4) <= 1:
            self._true_key = 'f'
            self._false_key = 'j'
        else:
            self._true_key = 'j'
            self._false_key = 'f'

        # Set the instruction
        self.math_guide = f'''左=【F】键                  右=【J】键

接下来将呈现一系列算数等式，请判断屏幕上
的等式是否成立。如果成立，则按【{self._true_key.upper()}】键；
如果不成立，则按【{self._false_key.upper()}】键。请尽快作答，思
考时间超过5秒则失去作答机会。

按【空格/SPACE】键继续'''
        # This will be used to record the information of math task
        self._math_idx = -1
        self.math_log = pd.DataFrame({'blockIndex': pd.Series(dtype='int'),
                                      'responseTime': pd.Series(dtype='float'),
                                      'ifCorrect': pd.Series(dtype='float')})

    def _get_camera(self) -> Camera:
        """Return a `Camera` instance for recording video and audio.

        The actual devices are matched by class parameters `microphone` and `camera`, so make sure you have set them, as well as `camFrameRate` and `camFrameSize`, before calling this method.
        The implementation of class `Camera` is achieved using `ffpyplayer`. You can also choose `opencv` as the `cameraLib` of `Camera`, but for now I don't know how to set the correct frame rate, frame size, and frame buffer using `opencv`.

        Note
        ------
        - The original parameter `bufferSecs` of `Camera` class is useless. When setting `ffpyplayer` as the `cameraLib`, what actualy determines the frame buffer size is `_bufferSecs` (default=0.5) in line 570 of `psychopy.hardware.camera`. I revised it to make `bufferSecs` of `Camera` work.
        - Due to an unknown bug of `MoviePy-1.0.3` library, the audio track cannot be merged with video track. Following this discussion (https://github.com/Zulko/moviepy/issues/1986), I made a modification to the `VideoClip.py` file within the `MoviePy` library. Specifically, I added the code below to the `write_video_file` method:

        >>> if fps is None:
        >>>     ps = self.fps
        """
        microphone = None
        availableMicrophones = Microphone.getDevices()
        for mic in availableMicrophones:
            if self._microphone in mic.deviceName:
                microphone = Microphone(
                    device=mic, streamBufferSecs=3, maxRecordingSize=72000)
        assert isinstance(
            microphone, Microphone), f"Microphone '{microphone}' not found!"
        # Set the camera
        camera = Camera(
            mic=microphone, frameRate=self._camFrameRate, frameSize=self._camFrameSize,
            # cameraLib='opencv', device=0,
            cameraLib='ffpyplayer', device=self._camera, bufferSecs=6)
        return camera

    def start(self) -> None:
        """Create and open an instance of `Window` for presenting experiment stimuli."""
        self._win = Window(units='norm',
                           fullscr=True,
                           color=[-1, -1, -1],
                           screen=self._stimuli_screen)
        self._mouse = event.Mouse(visible=False)

    def send_trigger(self, trigger: Union[int, List[int]]) -> None:
        """Send trigger(s) via `self._port`, or do nothing if `self._port` is None.

        The `trigger` should be int or list[int], and valid value(s) must be between 1-39 or between 48-57."""
        if self._port:
            if not isinstance(trigger, list):
                trigger = [trigger]
            for n in trigger:
                assert 1 <= n <= 39 or 48 <= n <= 57, 'Invalid trigger value(s): the trigger must be an integer between 1-39 or between 48-57.'
            self._port.write(bytes(trigger))

    def _flip_with_trigger(self, num: Union[int, List[int]]) -> None:
        """Send trigger via `self._port` while flip `self._win`."""
        self._win.callOnFlip(self.send_trigger, num)
        self._win.flip()

    def show_instruction(self,
                         text: str,
                         wait: Union[float, int, str, List[str], None],
                         pos: Tuple[float, float] = (0., 0.)) -> Union[List[str], None]:
        """Show instruction text in `self._win` and wait for a period.

        Parameters
        ------
        text : str
            The instruction to be shown.
        wait : float, int, str, list[str], or None
            If float or int, waiting time (in second). If str or list[str], key(s) waiting to be pressed. E.g., ['ctrl','q'] => show instruction until pressing `ctrl` and `q`. If None, constantly show instruction.
        pos : Tuple[float,float], default=(0.,0.), i.e., the center
            Position of the instruction in `self._win` with `unit` == 'norm'.

        Returns
        ------
        If `wait`=list[str], return the actually pressed key(s) as list[str]. Otherwise, None.
        """
        instruction = TextBox2(self._win,
                               font='Microsoft YaHei',
                               letterHeight=0.06,
                               units='norm',
                               alignment='center',
                               pos=pos,
                               text='')
        instruction.text = text
        instruction.draw()
        self._win.flip()
        if isinstance(wait, str):
            event.waitKeys(keyList=[wait])
        elif isinstance(wait, list):
            press = event.waitKeys(keyList=wait)
            return press
        elif isinstance(wait, (float, int)):
            core.wait(wait)

    def show_fixation(self,
                      wait: Union[float, int] = 5) -> None:
        """Show cross-fixation in the center of `self._win` and wait for a period.

        Parameters
        ------
        wait : float, int
            Waiting time (in second).
        """
        fixation = ShapeStim(self._win,
                             vertices=((0, -0.02), (0, 0.02),
                                       (0, 0), (-0.02, 0), (0.02, 0)),
                             closeShape=False,
                             lineWidth=4,
                             lineColor="white",
                             units='height')
        fixation.draw()
        self._win.flip()
        core.wait(wait)

    def _adjust_videoSize(self, video: MovieStim) -> None:
        """Adjust the displayed size of given video stimuli (instance of `MovieStim`) in place, acording to the actual size of `self._win`."""
        if self._win.size[0]/self._win.size[1] > video.videoSize[0]/video.videoSize[1]:
            desiredWidth = video.videoSize[0] * \
                self._win.size[1] / video.videoSize[1]
            desiredHeight = video.videoSize[1] * \
                self._win.size[1] / video.videoSize[1]
        else:
            desiredWidth = video.videoSize[0] * \
                self._win.size[0] / video.videoSize[0]
            desiredHeight = video.videoSize[1] * \
                self._win.size[0] / video.videoSize[0]
        video.size = (desiredWidth, desiredHeight)

    def _emotion_ratings(self, target: str = '刚才您实际体验到的情绪强度', prefix: Union[str, None] = None) -> Tuple[dict, list]:
        """Return the visual elements in Tuple[dict,list] for getting ratings on 8 discrete emotion items.

        Parameters
        ------
        target : str, default = '刚才您实际体验到的情绪强度'
            The to-be-rated target, e.g., "The emotion intensity that you actually experienced just now" (in Chinese).
        prefix : str or None, default = None
            Prefix string for emotion items when saving to `self.trials`.

        Returns
        ------
        scales : dict
            A dict mapping item names (str) to instances of `Slider`.
        labels : list
            A list containing instances of `TextBox2` for showing title, tick labels, item labels...

        Item Names
        ------
            - tenderness (of target)
            - joy (of target)
            - inspiration (of target)
            - amusement (of target)
            - sadness (of target)
            - fear (of target)
            - anger (of target)
            - disgust (of target)
        """
        scales = {}  # save all Slider instances for getting ratings
        labels = []  # save all TextBox2 instances for instructions
        # Prepare the instruction
        instruction1 = TextBox2(self._win,
                                text='请对'+target+'进行评分：',
                                font='Microsoft YaHei',
                                letterHeight=0.05,
                                alignment='center',
                                pos=(0, 0.9))
        labels.append(instruction1)
        # Prepare the rating items: 8 discrete emotions
        emotions_chn = ['温情', '高兴', '激励', '搞笑', '悲伤', '恐惧', '愤怒', '恶心']
        emotions_eng = ['tenderness', 'joy', 'inspiration', 'amusement',
                        'sadness', 'fear', 'anger', 'disgust']
        for i in range(8):
            scale = Slider(self._win,
                           markerColor='White',
                           ticks=(1, 2, 3, 4, 5, 6, 7),
                           pos=(0, 0.65-0.18*i),
                           size=(1, 0.03))
            emotion = TextBox2(self._win,
                               text=emotions_chn[i],
                               font='Microsoft YaHei',
                               letterHeight=0.05,
                               alignment='center',
                               pos=(-0.6, 0.65-0.18*i))
            scales[prefix+emotions_eng[i]
                   if prefix else emotions_eng[i]] = scale
            labels.append(emotion)
        degrees = ['完全不', '中等', '非常强']
        for i in range(3):
            degree = TextBox2(self._win,
                              text=degrees[i],
                              font='Microsoft YaHei',
                              letterHeight=0.05,
                              alignment='center',
                              pos=(-0.5+0.5*i, 0.75))
            labels.append(degree)
        # Reset the sliders
        for scale in scales.values():
            scale.reset()
        return scales, labels

    def _4item_ratings(self,
                       instructions: List[Union[str, None]],
                       itemNames: List[str],
                       tickLabels: List[Tuple[str]],
                       startValues: List[Union[int, None]]) -> Tuple[dict, list]:
        """Return the visual elements in Tuple[dict,list] for getting 4 ratings per page.

        Parameters
        ------
        instructions : list
            A list contains instruction strings above 4 scales, if `None` then no corresponding instruction.
        itemNames : list
            A list contains the name string of 4 rating items, used for saving in `self.trials`.
        tickLabels : list
            A list contains tuples ("minLabel","maxLabel") used for labeling 4 scales.
        startValues : list
            A list contains default start values (int) of 4 rating items, if `None` then no corresponding start value.

        Returns
        ------
        scales : dict
            A dict mapping item names (str) to instances of `Slider`.
        labels : list
            A list containing instances of `TextBox2` for showing title, tick labels, item labels...
        """
        scales = {}  # save all Slider instances for getting ratings
        labels = []  # save all TextBox2 instances for instructions

        for i in range(4):
            # Perpare the instruction
            if instructions[i] and len(instructions[i]) > 0:
                instruction = TextBox2(self._win,
                                       text=instructions[i],
                                       font='Microsoft YaHei',
                                       letterHeight=0.05,
                                       alignment='center',
                                       pos=(0, 0.83-i*0.4))
                labels.append(instruction)
            # Prepare the rating scale
            scale = Slider(self._win,
                           markerColor='White',
                           ticks=(1, 2, 3, 4, 5, 6, 7),
                           startValue=startValues[i],
                           pos=(0, 0.65-i*0.4),
                           size=(1, 0.03))
            scales[itemNames[i]] = scale
            # Prepare the tick labels
            degrees = tickLabels[i]
            assert len(degrees) == 2
            for j in range(2):
                degree = TextBox2(self._win,
                                  text=degrees[j],
                                  font='Microsoft YaHei',
                                  letterHeight=0.05,
                                  alignment='center',
                                  pos=(-0.5+1*j, 0.75-i*0.4))
                labels.append(degree)

        # Reset the sliders
        for scale in scales.values():
            scale.reset()
        return scales, labels

    def _affect_ratings(self, stim_type: Union[str, None] = '这段视频') -> Tuple[dict, list]:
        """Return the visual elements in Tuple[dict,list] for getting ratings on 4 affect items.

        Parameters
        ------
        stim_type : str, default='这段视频'
            Name of the to-be-rated stimuli.

        Returns
        ------
        scales : dict
            A dict mapping item names (str) to instances of `Slider`.
        labels : list
            A list containing instances of `TextBox2` for showing title, tick labels, item labels...

        Item Names
        ------
            - valence (of self)
            - arousal (of self)
            - liking (to stimuli)
            - familiarity (to stimuli)
        """
        return self._4item_ratings(instructions=['总的来说，您的主观感受是：', None,
                                                 f'您对{stim_type}的看法是：', None],
                                   itemNames=['valence', 'arousal',
                                              'liking', 'familiarity'],
                                   tickLabels=[('消极', '积极'),
                                               ('平静', '强烈'),
                                               ('不喜欢', '喜欢'),
                                               ('不熟悉', '熟悉')],
                                   startValues=[4, 1, 4, 1])

    def _buttons(self) -> Tuple[TextBox2, TextBox2, TextBox2, list, list, list]:
        """Return the botton elements and positions for page control.

        Returns
        ------
        next_btn : TextBox2
        return_btn : TextBox2
        confirm_btn : TextBox2
        left_pos : tuple
        middle_pos : tuple
        right_pos : tuple
        """
        left_pos = [-0.3, -0.85]
        middle_pos = [0, -0.85]
        right_pos = [0.3, -0.85]
        next_btn = TextBox2(self._win,
                            text='下一页',
                            font='Microsoft YaHei',
                            letterHeight=0.05,
                            color='black',
                            size=[0.2, 0.15],
                            fillColor='white',
                            alignment='center',
                            pos=[0, 0])
        return_btn = TextBox2(self._win,
                              text='上一页',
                              font='Microsoft YaHei',
                              letterHeight=0.05,
                              color='black',
                              size=[0.2, 0.15],
                              fillColor='white',
                              alignment='center',
                              pos=[0, 0])
        confirm_btn = TextBox2(self._win,
                               text='提交',
                               font='Microsoft YaHei',
                               letterHeight=0.05,
                               color='black',
                               size=[0.2, 0.15],
                               fillColor='white',
                               alignment='center',
                               pos=[0, 0])
        return next_btn, return_btn, confirm_btn, left_pos, middle_pos, right_pos

    def _rating_control(self, pages: List[Tuple[dict, list]], log: bool = True) -> None:
        """Page flow control when getting ratings.

        The subject can freely switch to different pages and modify the ratings until he/she clicks on the `确认` (Confirm) button on the final page. The buttons on each page appear only after all the `sliders` on the current page have been clicked.

        Parameters
        ------
        pages : List[Tuple]
            A list containing different pages' elements. Each page must be a tuple containing a dict (mapping item names to `Slider`) and a list (all other `TextBox2` instances).
        log : bool, default=True
            The flag controlling whether to save rating results in `self.trials`.
        """
        next_btn, return_btn, confirm_btn, left_pos, middle_pos, right_pos = self._buttons()
        n_pages = len(pages)
        page_flag = 0

        def _draw_and_check_scale(page) -> bool:
            scale_complete = True
            for label in page[1]:
                label.draw()
            for scale in page[0].values():
                scale.draw()
                if scale.getRating() is None:
                    scale_complete = False
            return scale_complete

        def _record_ratings(page):
            for item, scale in page[0].items():
                self.trial_log.at[self.trial_idx, item] = (
                    scale.getRating() - 1) * 100 / 6  # Convert [1,7] to [0,100]

        # Four kinds of button-combinations
        self._mouse.setVisible(True)
        confirmed = False
        while not confirmed:
            if n_pages == 1:
                complete = _draw_and_check_scale(pages[page_flag])
                if complete:
                    confirm_btn.pos = middle_pos
                    confirm_btn.draw()
                    # Save ratings and end this session
                    if self._mouse.isPressedIn(confirm_btn):
                        if log:
                            _record_ratings(pages[page_flag])
                        self._mouse.setVisible(False)
                        confirmed = True
            elif n_pages > 1:
                if page_flag == 0:
                    complete = _draw_and_check_scale(pages[page_flag])
                    if complete:
                        next_btn.pos = middle_pos
                        next_btn.draw()
                        # Save ratings and move to next page
                        if self._mouse.isPressedIn(next_btn):
                            if log:
                                _record_ratings(pages[page_flag])
                            page_flag += 1
                elif page_flag < (n_pages-1):
                    complete = _draw_and_check_scale(pages[page_flag])
                    if complete:
                        return_btn.pos, next_btn.pos = left_pos, right_pos
                        return_btn.draw()
                        next_btn.draw()
                        # Save ratings and move to next page
                        if self._mouse.isPressedIn(next_btn):
                            if log:
                                _record_ratings(pages[page_flag])
                            page_flag += 1
                        # Save ratings and move to last page
                        elif self._mouse.isPressedIn(return_btn):
                            if log:
                                _record_ratings(pages[page_flag])
                            page_flag -= 1
                elif page_flag == (n_pages-1):
                    complete = _draw_and_check_scale(pages[page_flag])
                    if complete:
                        return_btn.pos, confirm_btn.pos = left_pos, right_pos
                        return_btn.draw()
                        confirm_btn.draw()
                        if self._mouse.isPressedIn(return_btn):
                            if log:
                                _record_ratings(pages[page_flag])
                            page_flag -= 1
                        elif self._mouse.isPressedIn(confirm_btn):
                            if log:
                                _record_ratings(pages[page_flag])
                            self._mouse.setVisible(False)
                            confirmed = True
            # Refresh the screen
            self._win.flip()

    def _lcm(self, x: int, y: int) -> int:
        """Return the least common denominator of the given `x` and `y`."""
        greater = max(x, y)
        while True:
            if ((greater % x == 0) and (greater % y == 0)):
                lcm = greater
                break
            greater += 1
        return lcm

    def do_math(self,
                n: int = 10,
                maxTime: float = 5.0,
                log: bool = True) -> None:
        """Let the subject do the equation judgment task.

        The subject needs to determine whether the equation presented on `self._win` is true or not by pressing the `F` key or the `J` key. The correspondence between the keys and the different judgments was determined by the parity of `self._subject['id']`. The result of the judgment is fed back to the subject by presenting a second of text after the key is pressed.

        Parameters
        ------
        n : int, default=10
            Number of to-be-judged equations. (default: 10)
        maxTome : float, default=0.5
            The time limit for waiting for a response, after which a 1-second `超时` (timeout) is displayed.
        log : bool, default=True
            The flag controlling whether to save the response time (`np.nan` if time out) and judgment result (`1` if correct, `0` if incorrect) in `self.trials`.
        """
        # Prepare visual elements
        if (self._subject['id'] % 4) <= 1:
            true_pos = (-0.5, -0.5)
            false_pos = (0.5, -0.5)
        else:
            true_pos = (0.5, -0.5)
            false_pos = (-0.5, -0.5)
        true_reminder = TextBox2(self._win,
                                 font='Microsoft YaHei',
                                 letterHeight=0.1,
                                 alignment='center',
                                 text=f'成立按【{self._true_key.upper()}】键',
                                 pos=true_pos)
        false_reminder = TextBox2(self._win,
                                  font='Microsoft YaHei',
                                  letterHeight=0.1,
                                  alignment='center',
                                  text=f'不成立按【{self._false_key.upper()}】键',
                                  pos=false_pos)
        center_message = TextBox2(self._win,
                                  font='Microsoft YaHei',
                                  letterHeight=0.1,
                                  alignment='center',
                                  text='')

        # Prepare math equations
        equation_true = [True]*(n//2)+[False]*(n-(n//2))
        do_multiply = [True]*(n//2)+[False]*(n-(n//2))
        random.Random(int(datetime.now().strftime('%S'))+1
                      ).shuffle(equation_true)
        random.Random(int(datetime.now().strftime('%S')) +
                      2).shuffle(do_multiply)
        a = random.Random(int(datetime.now().strftime('%S'))+3
                          ).choices(range(2, 10), k=n)  # 11, 20
        b = random.Random(int(datetime.now().strftime('%S'))+4
                          ).choices(range(2, 10), k=n)
        c = random.Random(int(datetime.now().strftime('%S'))+5
                          ).choices(range(1, 31), k=n)  # 1, 51
        d = []
        for i in range(n):
            if do_multiply[i]:  # A*B+C=D
                if equation_true[i]:
                    d.append(int(a[i]*b[i]+c[i]))
                else:
                    d.append(int(a[i]*b[i]+c[i]+(-1) **
                             (random.Random(i).choice((1, 2)))*10))
            else:  # A/B+C=D
                a[i] = self._lcm(b[i], a[i])
                if equation_true[i]:
                    d.append(int(a[i]/b[i]+c[i]))
                else:
                    d.append(int(a[i]/b[i]+c[i]+(-1) **
                             (random.Random(i).choice((1, 2)))*10))

        # Task start
        timer = core.Clock()
        for i in range(n):
            if log:
                self._math_idx += 1
                self.math_log.at[self._math_idx, 'blockIndex'] = self.block_idx
            # Show the equation
            true_reminder.draw()
            false_reminder.draw()
            if do_multiply[i]:
                center_message.text = f'{a[i]} * {b[i]} + {c[i]} = {d[i]}'
            else:
                center_message.text = f'{a[i]} / {b[i]} + {c[i]} = {d[i]}'
            center_message.draw()
            self._win.callOnFlip(timer.reset)
            self._win.flip()
            # Get the response
            key = event.waitKeys(maxWait=maxTime,
                                 keyList=[self._true_key, self._false_key])
            rt = timer.getTime()
            # Check the response and feedback
            if key is None:  # Time out
                center_message.text = '超时'
                if log:
                    self.math_log.at[self._math_idx, 'responseTime'] = np.nan
                    self.math_log.at[self._math_idx, 'ifCorrect'] = np.nan
            elif (key[0] == self._true_key) ^ equation_true[i]:  # Wrong response
                center_message.text = '作答错误'
                if log:
                    self.math_log.at[self._math_idx, 'responseTime'] = rt
                    self.math_log.at[self._math_idx, 'ifCorrect'] = 0.
            else:  # Right response
                center_message.text = '作答正确'
                if log:
                    self.math_log.at[self._math_idx, 'responseTime'] = rt
                    self.math_log.at[self._math_idx, 'ifCorrect'] = 1.
            center_message.draw()
            self._win.flip()
            core.wait(1)

        # Overwrite the local file to update results
        if log:
            self.math_log.to_csv(self._mathFile, index=False,
                                 index_label=False, mode='w')

    def _end(self):
        """The routine at the end of the experiment:
            - display the mouse
            - close `self._win` and `self._port`"""
        self._mouse.setVisible(True)
        self._win.close()
        self._end_time = datetime.today().strftime('%H:%M:%S')
        if self._port:
            self._port.close()


class Experiment1(Experiment):
    """Class for experiment 1 (watching videos).

    Parameters
    ------
    exp_dir : str
        The folder path that contains all nessecary files supporting this experiment.
    stimuli_screen, observe_screen : int
        Indeces of the screens for showing stimuli, and for you to observe computer activities.
    port : serial.Serial or None, default=None
        A port instance for sending trigger to the fNIRS device.

    Attributes
    ------
    math_guide : str
        Instruction for equation judgment task.
    math_log : pd.DataFrame
        Columns include `blockIndex`, `responseTime`, and `ifCorrect`.
    block_idx, trial_idx: int
        Changeable index of the current block and the current trial. 1 block = 4 trials = 4 video clips.
    blocks : List[List[int]]
        A nested list containing 7 blocks and 4 video indeces within each block.
    trial_log : pd.DataFrame
        Columns include `videoIndex`, `videoEmotion`, `percCompleted`, `anger`, `disgust`, `fear`, `sadness`, `amusement`, `inspiration`, `joy`, `tenderness`, `valence`, `arousal`, `liking`, and `familiarity`.
    welcome_guide : str
        The main instruction of this experiment.
    practice_guide : str
        The instruction for practice session.

    Methods
    ------
    start()
        Create and open an instance of `Window` for presenting experiment stimuli.
    send_trigger()
        Send trigger(s) via `self._port`, or do nothing if `self._port` is None.
    show_instruction(text, wait, pos=(0.,0.)) -> List[str] or None
        Show instruction text in `self._win` and wait for a period.
    show_fixation(wait=5)
        Show cross-fixation in the center of `self._win` and wait for a period.
    do_math(n=10, maxTime=5.0, log=True)
        Let the subject do the equation judgment task.
    play_video(video_idx, log=True, skip='escape')
        Play the mp4 file specified by the given `video_idx`.
    get_ratings(log=True)
        Get the self-reported ratings about the video that just played.
    end()
        End this experiment.
    """

    def __init__(self, exp_dir, stimuli_screen, observe_screen, port: Union[serial.Serial, None] = None):
        super().__init__(exp_dir=exp_dir, stimuli_screen=stimuli_screen,
                         observe_screen=observe_screen, exp_id='exp1', port=port)
        self._video_dir = os.path.join(exp_dir, 'stimuli_exp1')
        # '愤怒','恶心','恐惧','悲伤','中性','搞笑','激励','高兴','温情'
        self._video_emotions = {1: 'anger', 2: 'anger', 3: 'anger',
                                4: 'disgust', 5: 'disgust', 6: 'disgust',
                                7: 'fear', 8: 'fear', 9: 'fear',
                                10: 'sadness', 11: 'sadness', 12: 'sadness',
                                13: 'neutral', 14: 'neutral', 15: 'neutral', 16: 'neutral',
                                17: 'amusement', 18: 'amusement', 19: 'amusement',
                                20: 'inspiration', 21: 'inspiration', 22: 'inspiration',
                                23: 'joy', 24: 'joy', 25: 'joy',
                                26: 'tenderness', 27: 'tenderness', 28: 'tenderness'}
        self._set_subject_info()
        self._set_math_keys()
        self._set_experiment_info()
        # Randomize the order of videos within and between blocks
        # 1 block = 4 trials with the same valence
        # 1 trial = 1 movie clip
        self.blocks = []
        self.blocks.extend(self._randomize_videos(
            1, 12, self._subject['id']+100))  # 3 negative blocks
        self.blocks.extend(self._randomize_videos(
            13, 16, self._subject['id']+200))  # 1 neutral blocks
        self.blocks.extend(self._randomize_videos(
            17, 28, self._subject['id']+300))  # 3 positive blocks
        random.Random(int(self._subject['id'])).shuffle(
            self.blocks)  # List[List[int]]
        # This will be used to record the information of each trial
        self.trial_log = pd.DataFrame({'videoIndex': pd.Series(dtype='int'),
                                       'videoEmotion': pd.Series(dtype='str'),
                                       'percCompleted': pd.Series(dtype='float'),
                                       'anger': pd.Series(dtype='float'),
                                       'disgust': pd.Series(dtype='float'),
                                       'fear': pd.Series(dtype='float'),
                                       'sadness': pd.Series(dtype='float'),
                                       'amusement': pd.Series(dtype='float'),
                                       'inspiration': pd.Series(dtype='float'),
                                       'joy': pd.Series(dtype='float'),
                                       'tenderness': pd.Series(dtype='float'),
                                       'valence': pd.Series(dtype='float'),
                                       'arousal': pd.Series(dtype='float'),
                                       'liking': pd.Series(dtype='float'),
                                       'familiarity': pd.Series(dtype='float')})
        self.welcome_guide = '''欢迎参加本实验！

接下来将播放一系列情绪诱发视频，可能唤起
积极、消极或中性情绪。请仔细观看视频，并
在每段视频结束后反馈您自身的情绪感受。如
果您感到强烈不适，可以按【退出/ESC】键跳
过正在播放的视频。

按【空格/SPACE】键继续'''
        self.practice_guide = '''为了帮助您更好地熟悉实验流程，接下来请练
习观看视频和评分，然后完成一轮数学等式判
断任务。练习过程中请您通过键盘上的滚轮将
音量调节至您觉得舒适的程度，正式实验中尽
量不要调节音量。如果在练习环节有任何疑问，
请随时向主试问询。

按【空格/SPACE】键继续'''

    def _randomize_videos(self, start_idx: int, end_idx: int, seed: int) -> List[List[int]]:
        """Randomize the order of video indeces with the same valence using given random `seed`, and returns them in nested list."""
        videos = list(range(start_idx, end_idx+1))
        random.Random(seed).shuffle(videos)
        trials_in_blocks = [videos[i:i+4] for i in range(0, len(videos), 4)]
        return trials_in_blocks

    def play_video(self,
                   video_idx: int,
                   log: bool = True,
                   skip: str = 'escape'):
        """Play the mp4 file specified by the given `video_idx`.

        Parameters
        ------
        video_idx : int
            The index must be between 1-28.
        log : bool, default=True
            The flag controlling whether to treat this play as a formal trial. If True, send trigger and record `videoIndex`, `videoEmotion`, and `percCompleted` in `self.trials`.
        skip : str, default='escape'
            A key that when pressed skips the currently playing video.
        """
        if log:
            self.trial_idx += 1
            self.trial_log.at[self.trial_idx, 'videoIndex'] = video_idx
            self.trial_log.at[self.trial_idx,
                              'videoEmotion'] = self._video_emotions[video_idx]
            self.send_trigger(video_idx)
        video_file = os.path.join(self._video_dir, f'{video_idx}.mp4')
        video = MovieStim(self._win, video_file, name=f'video{video_idx}')
        self._adjust_videoSize(video)
        # Display
        if log:
            self._win.callOnFlip(self.send_trigger, 51)
        while not video.isFinished:
            video.draw()
            self._win.flip()
            if event.getKeys(keyList=[skip]):
                if log:
                    self.send_trigger(50)
                video.pause()
                break
        video.stop()
        if log:
            self.send_trigger(52)
            self.trial_log.at[self.trial_idx,
                              'percCompleted'] = video.getPercentageComplete()

    def get_ratings(self,
                    log: bool = True):
        """Get the self-reported ratings about the video that just played.

        Parameters
        ------
        log : bool, default=True
            The flag controlling whether to treat this play as a formal trial. If True, record ratings in `self.trials`.

        Rating Items
        ------
            - tenderness (of self)
            - joy (of self)
            - inspiration (of self)
            - amusement (of self)
            - sadness (of self)
            - fear (of self)
            - anger (of self)
            - disgust (of self)
            - valence (of self)
            - arousal (of self)
            - liking (to stimuli)
            - familiarity (to stimuli)
        """
        # Get elements in different pages
        pages = [self._emotion_ratings(),
                 self._affect_ratings(stim_type='这段视频')]

        # Draw pages and get ratings
        self._rating_control(pages, log)

        # Overwrite the local file to update ratings
        if log:
            self.trial_log.to_csv(self._ratingFile, index=False,
                                  index_label=False, mode='w')

    def end(self):
        """The routine at the end of the experiment:
            - display the mouse
            - close `self._win` and `self._port`
            - save `self.trials` and `self.math`"""
        self._end()
        # Final update the local files
        self.trial_log.to_csv(self._ratingFile, index=False,
                              index_label=False, mode='w')
        self.math_log.to_csv(self._mathFile, index=False,
                             index_label=False, mode='w')


class Experiment2(Experiment):
    """Class for experiment 2 (recalling/telling memory).

    Parameters
    ------
    exp_dir : str
        The folder path that contains all nessecary files supporting this experiment.
    stimuli_screen, observe_screen : int
        Indeces of the screens for showing stimuli, and for you to observe computer activities.
    microphone : str or None
        Name of the microphne for recording audios.
    camera : str or None
        Name of the camera for recording videos.
    camFrameSize : tuple or None
        Frame size of the recorded video.
    camFrameRate : int or None
        rame rate of the recorded video.
    port : serial.Serial or None, default=None
        A port instance for sending trigger to the fNIRS device.

    Attributes
    ------
    math_guide : str
        Instruction for equation judgment task.
    math_log : pd.DataFrame
        Columns include `blockIndex`, `responseTime`, and `ifCorrect`.
    block_idx, trial_idx: int
        Changeable index of the current block and the current trial.
    blocks : List[List[str]]
        A nested list containing a positive block, a neutral block, and a negative block. The order of blocks and the order of emotions in each block are randomized.
    trial_log : pd.DataFrame
        Columns include `targetEmotion`, `recallCompletedPerc`, `tellCompletedPerc`, `anger`, `disgust`, `fear`, `sadness`, `amusement`, `inspiration`, `joy`, `tenderness`, `valence`, `arousal`, `liking`, and `familiarity`.
    welcome_guide : str
        The main instruction of this experiment.
    practice_guide : str
        The instruction for practice session.
    tell_guide : str
        The instruction for telling memory.

    Methods
    ------
    start()
        Create and open an instance of `Window` for presenting experiment stimuli.
    send_trigger()
        Send trigger(s) via `self._port`, or do nothing if `self._port` is None.
    show_instruction(text, wait, pos=(0.,0.)) -> List[str] or None
        Show instruction text in `self._win` and wait for a period.
    show_fixation(wait=5)
        Show cross-fixation in the center of `self._win` and wait for a period.
    do_math(n=10, maxTime=5.0, log=True)
        Let the subject do the equation judgment task.
    recall_guide(emotion)
        Return recall instruction for the given `emotion`.
    recall_task(duration, prepare=None, log=True, skip='escape')
        Countdown for the subject's silent recall.
    recall_check(log=True)
        Ask the subject if he or she successfully recalled an event.
    get_ratings(log=True)
        Get the self-reported ratings about the memory that just recalled.
    tell_task(duration, prepare=None, log=True, save=True, emotion=None, skip='escape')
        Count down for the subject's telling. Meanwhile, record a video (optional).
    end()
        End this experiment.
    """

    def __init__(self, exp_dir, stimuli_screen, observe_screen,
                 microphone: str,
                 camera: str,
                 camFrameSize: Tuple[int],
                 camFrameRate: int,
                 port: Union[serial.Serial, None] = None) -> None:
        super().__init__(exp_dir=exp_dir,
                         exp_id='exp2',
                         stimuli_screen=stimuli_screen,
                         observe_screen=observe_screen,
                         port=port,
                         microphone=microphone,
                         camera=camera,
                         camFrameSize=camFrameSize,
                         camFrameRate=camFrameRate)
        self._set_subject_info()
        self._set_math_keys()
        self._set_experiment_info()
        # Randomize the order of videos within and between blocks
        # 1 block = 3 trials with the same valence
        # 1 trial = 1 emotional event
        pos_emotions = ['tenderness', 'joy', 'inspiration', 'amusement']
        neg_emotions = ['sadness', 'fear', 'disgust', 'anger']
        random.Random(self._subject['id']+400).shuffle(pos_emotions)
        random.Random(self._subject['id']+500).shuffle(neg_emotions)
        if (self._subject['id'] % 2) == 0:
            self.blocks = [pos_emotions, ['neutral'], neg_emotions]
        else:
            self.blocks = [neg_emotions, ['neutral'], pos_emotions]
        # This will be used to record the information of each trial
        self.trial_log = pd.DataFrame({'targetEmotion': pd.Series(dtype='str'),
                                    'recallCompletedPerc': pd.Series(dtype='float'),
                                    'tellCompletedPerc': pd.Series(dtype='float'),
                                    'anger': pd.Series(dtype='float'),
                                    'disgust': pd.Series(dtype='float'),
                                    'fear': pd.Series(dtype='float'),
                                    'sadness': pd.Series(dtype='float'),
                                    'amusement': pd.Series(dtype='float'),
                                    'inspiration': pd.Series(dtype='float'),
                                    'joy': pd.Series(dtype='float'),
                                    'tenderness': pd.Series(dtype='float'),
                                    'valence': pd.Series(dtype='float'),
                                    'arousal': pd.Series(dtype='float'),
                                    'liking': pd.Series(dtype='float'),
                                    'familiarity': pd.Series(dtype='float')})
        # Set the main instructions
        self.welcome_guide = '''欢迎参加本实验！

接下来您将回忆一系列您真实经历过的事件，
它们可能唤起积极、消极或中性情绪。请充分
回忆，之后将这些事讲述出来。我们会对这一
过程进行录像，但绝不会将其用于学术研究以
外的用途。如果您感到强烈不适，随时可以按
【退出/ESC】键跳过回忆/讲述任务。

按【空格/SPACE】键继续'''
        self.practice_guide = '''为了帮助您更好地熟悉实验流程，接下来请练
习一个完整的回忆任务，然后练习一轮数学等
式判断任务。练习过程中如有任何疑问，请随
时向主试问询。

按【空格/SPACE】键继续'''
        self.tell_guide = '''接下来请注视屏幕上呈现的十字，想象面前有
几位朋友对您的经历很感兴趣。请尽可能放松，
用您最真实的状态讲述刚才回忆的这件事以及
您的感受。

此任务将持续180秒，请保持讲述一直到结束。
若您觉得确实需要提前结束，按【退出/ESC】键。

按【空格/SPACE】键开始'''
        self._check_guide = '''您是否回想起了一件具体的事？

若【是】，按【空格/SPACE】键继续
若【否】，按【退出/ESC】键跳过
'''

    def recall_guide(self, emotion: str) -> str:
        """Return recall instruction of the given `emotion`."""
        if emotion == 'neutral':
            text = f'''请在脑海中回想一件您曾经或最近经历过的事，
这件事在当时并没有激发您任何的感受。您无
需发出任何声音，只需要回忆起这件事的细节，
尽可能将自己代入其中，重新去体验这段经历。

您将有90秒完成此任务，按【空格/SPACE】键开始'''
        else:
            text = f'''请在脑海中回想一件您曾经或最近经历过的事，
这件事在当时令您感到非常的【{self._target_emotions[emotion]}】，您无
需发出任何声音，只需要回忆起这件事的细节，
尽可能将自己代入其中，重新去体验这段经历。

您将有90秒完成此任务，按【空格/SPACE】键开始'''
        return text

    def recall_task(self,
                    duration: Union[float, int],
                    prepare: Union[float, int, None] = None,
                    log: bool = True,
                    skip: str = 'escape') -> None:
        """Countdown for the subject's silent recall.

        Parameters
        ------
        duration : float or int
            Max time limit of recalling.
        prepare : float, int, or None, default=None
            If not None, countdown `prepare` time (in seconds) before recalling.
        log : bool, default=True
            The flag controlling whether to treat this recall as a formal trial. If True, send trigger and record 'recallCompletedPerc' in `self.trials`.
        skip : str, default='escape'
            A key that when pressed ends the currently recalling.
        """
        centerMessage = TextBox2(self._win,
                                 font='Microsoft YaHei',
                                 letterHeight=0.15,
                                 alignment='center',
                                 pos=(0, 0),
                                 text='')
        timeReminder = TextBox2(self._win,
                                font='Microsoft YaHei',
                                letterHeight=0.07,
                                alignment='center',
                                pos=(0, 0.8),
                                text=f'剩余{math.ceil(duration)}秒')
        enddingSound = Sound(os.path.join(self._exp_dir, 'dee.mp3'))
        if prepare:
            preTimer = core.CountdownTimer(prepare)
            while preTimer.getTime() > 0:
                centerMessage.text = str(math.ceil(preTimer.getTime()))
                centerMessage.draw()
                timeReminder.draw()
                self._win.flip()
        # Task start
        remainTime = 0
        centerMessage.text = '静默回忆'
        timer = core.CountdownTimer(duration)
        if log:
            self.send_trigger(53)
        while timer.getTime() > 0:
            timeReminder.text = f'剩余{math.ceil(timer.getTime())}秒'
            timeReminder.draw()
            centerMessage.draw()
            self._win.flip()
            if event.getKeys(keyList=[skip]):
                if log:
                    self.send_trigger(50)
                remainTime = timer.getTime()
                break
        if log:
            self.send_trigger(54)
            self.trial_log.at[self.trial_idx,
                           'recallCompletedPerc'] = 100*(1 - remainTime/duration)
        enddingSound.play()
        centerMessage.text = '回忆结束'
        centerMessage.draw()
        self._win.flip()
        core.wait(3)

    def recall_check(self, log: bool = True) -> bool:
        """Ask the subject if he or she successfully recalled an event.

        Parameters
        ------
        log : bool, default=True
            The flag controlling whether to treat this recall as a formal trial. If `log`=True and the subject didn't recalled an event, modify both 'recallCompletedPerc' and 'tellCompletedPerc' in `self.trials` as 0.

        Returns
        ------
        If the subject successfully recalled an event, True. Otherwise, False.
        """
        press = self.show_instruction(
            text=self._check_guide, wait=['space', 'escape'])
        assert len(press) == 1
        if press[0] == 'space':
            return True
        elif press[0] == 'escape':
            if log:
                self.trial_log.at[self.trial_idx, 'recallCompletedPerc'] = 0
                self.trial_log.at[self.trial_idx, 'tellCompletedPerc'] = 0
                self.trial_log.to_csv(self._ratingFile, index=False,
                                   index_label=False, mode='w')
            return False

    def get_ratings(self,
                    log: bool = True):
        """Get the self-reported ratings about the memory that just recalled.

        Parameters
        ------
        log : bool, default=True
            The flag controlling whether to treat this recall as a formal trial. If True, record ratings in `self.trials`.

        Rating Items
        ------
            - tenderness (of self)
            - joy (of self)
            - inspiration (of self)
            - amusement (of self)
            - sadness (of self)
            - fear (of self)
            - anger (of self)
            - disgust (of self)
            - valence (of self)
            - arousal (of self)
            - liking (to memory)
            - familiarity (to memory)
        """
        # Get elements in different pages
        pages = [self._emotion_ratings(),
                 self._affect_ratings(stim_type='这段经历')]

        # Draw pages and get ratings
        self._rating_control(pages, log)

        # Overwrite the local file to update ratings
        if log:
            self.trial_log.to_csv(self._ratingFile, index=False,
                               index_label=False, mode='w')

    def tell_task(self,
                  duration: Union[float, int],
                  prepare: Union[float, int, None] = None,
                  log: bool = True,
                  save: bool = True,
                  emotion: Union[str, None] = None,
                  skip: str = 'escape') -> None:
        """Count down for the subject's telling. Meanwhile, record a video (optional).

        Parameters
        ------
        duration : float or int
            Max time limit of telling.
        prepare : float, int, or None, default=None
            If not None, countdown `prepare` time (in seconds) before telling.
        log : bool, default=True
            The flag controlling whether to treat this telling as a formal trial. If True, send trigger and record 'tellCompletedPerc' in `self.trials`.
        save : bool, default=True
            The flag controlling whether to record and save a video of this telling.
        emotion : str or None, default=None
            If `save`=True, use `emotion` to name the saved mp4 file.
        skip : str, default='escape'
            A key that when pressed ends the currently telling.

        Note
        ------
        The `save` feature is very demanding on both cpu and memory, and creates a thread block for a certain amount of time after the telling is over (3 minutes of recording -> about 2 minutes of blocking). Therefore, treat the blocking time as a break session.

        See Also
        ------
        Method `_get_camera()`.
        """
        centerMessage = TextBox2(self._win,
                                 font='Microsoft YaHei',
                                 letterHeight=0.15,
                                 alignment='center',
                                 pos=(0, 0),
                                 text='')
        timeReminder = TextBox2(self._win,
                                font='Microsoft YaHei',
                                letterHeight=0.07,
                                alignment='center',
                                pos=(0, 0.8),
                                text=f'剩余{math.ceil(duration)}秒')
        enddingSound = Sound(os.path.join(self._exp_dir, 'dee.mp3'))
        camera = self._get_camera()
        if save:
            assert emotion, "The output recording file must be named with parameter 'emotion'."
        # Count down before task
        if prepare:
            preTimer = core.CountdownTimer(prepare)
            while preTimer.getTime() > 0:
                centerMessage.text = str(math.ceil(preTimer.getTime()))
                centerMessage.draw()
                timeReminder.draw()
                self._win.flip()
        # Task start
        recordTime = duration
        centerMessage.text = '+'
        camera.open()
        camera.record()
        if log:
            self.send_trigger(55)
        while camera.recordingTime < duration:
            camera.update()
            timeReminder.text = f'剩余{math.ceil(duration - camera.recordingTime)}秒'
            timeReminder.draw()
            centerMessage.draw()
            self._win.flip()
            if event.getKeys(keyList=[skip]):
                if log:
                    self.send_trigger(50)
                recordTime = camera.recordingTime
                break
        if log:
            self.send_trigger(56)
            self.trial_log.at[self.trial_idx,
                           'tellCompletedPerc'] = 100*(recordTime/duration)
        enddingSound.play()
        camera.stop()
        camera.close()
        centerMessage.text = '讲述结束\n\n请从回忆中放松\n并休息一会\n\n（不要按任何键）'
        centerMessage.draw()
        self._win.flip()
        if save:
            camera.save(os.path.join(
                self._save_dir, f'{self._exp_id}_{self._date}{self._start_time}_{emotion}.mp4'), mergeAudio=True)

    def end(self):
        """The routine at the end of the experiment:
            - display the mouse
            - close `self._win` and `self._port`
            - save `self.trials` and `self.math`"""
        self._end()
        # Final update the local files
        self.trial_log.to_csv(self._ratingFile, index=False,
                           index_label=False, mode='w')
        self.math_log.to_csv(self._mathFile, index=False,
                             index_label=False, mode='w')


class Experiment3(Experiment):
    """Class for experiment 3 (acting given script).

    Parameters
    ------
    exp_dir : str
        The folder path that contains all nessecary files supporting this experiment.
    stimuli_screen, observe_screen : int
        Indeces of the screens for showing stimuli, and for you to observe computer activities.
    microphone : str or None
        Name of the microphne for recording audios.
    camera : str or None
        Name of the camera for recording videos.
    camFrameSize : tuple or None
        Frame size of the recorded video.
    camFrameRate : int or None
        rame rate of the recorded video.
    port : serial.Serial or None, default=None
        A port instance for sending trigger to the fNIRS device.

    Attributes
    ------
    math_guide : str
        Instruction for equation judgment task.
    math_log : pd.DataFrame
        Columns include `blockIndex`, `responseTime`, and `ifCorrect`.
    block_idx, trial_idx: int
        Changeable index of the current block and the current trial.
    blocks : List[List[str]]
        A nested list containing a positive block, a neutral block, and a negative block. The order of blocks and the order of emotions in each block are randomized.
    trial_log : pd.DataFrame
        Columns include 'targetEmotion', 'actCompletedPerc', and 28 rating items.
    welcome_guide : str
        The main instruction of this experiment.
    practice_guide : str
        The instruction for practice session.
    act_guide : str
        The instruction for acting.

    Methods
    ------
    start()
        Create and open an instance of `Window` for presenting experiment stimuli.
    send_trigger()
        Send trigger(s) via `self._port`, or do nothing if `self._port` is None.
    show_instruction(text, wait, pos=(0.,0.)) -> List[str] or None
        Show instruction text in `self._win` and wait for a period.
    show_fixation(wait=5)
        Show cross-fixation in the center of `self._win` and wait for a period.
    do_math(n=10, maxTime=5.0, log=True)
        Let the subject do the equation judgment task.
    rehersal_guide(emotion)
        Return rehersal instruction for the given `emotion`.
    show_script(emotion, log=True)
        Display the script of the given emotion and record it if log.
    act_task(duration, prepare=None, log=True, save=True, emotion=None, skip='escape')
        Count down for the subject's acting. Meanwhile, record a video (optional).
    get_ratings(log=True)
        Get the self-reported ratings about the acting task.
    end()
        End this experiment.
    """

    def __init__(self, exp_dir, stimuli_screen, observe_screen,
                 microphone: str,
                 camera: str,
                 camFrameSize: Tuple[int],
                 camFrameRate: int,
                 port: Union[serial.Serial, None] = None) -> None:
        super().__init__(exp_dir=exp_dir,
                         exp_id='exp3',
                         stimuli_screen=stimuli_screen,
                         observe_screen=observe_screen,
                         port=port,
                         microphone=microphone,
                         camera=camera,
                         camFrameSize=camFrameSize,
                         camFrameRate=camFrameRate)
        self._set_subject_info()
        self._set_math_keys()
        self._set_experiment_info()
        # Randomize the order of videos within and between blocks
        # 1 block = 3 trials with the same valence
        # 1 trial = 1 emotional event
        pos_emotions = ['tenderness', 'joy', 'inspiration', 'amusement']
        neg_emotions = ['sadness', 'fear', 'disgust', 'anger']
        random.Random(self._subject['id']+600).shuffle(pos_emotions)
        random.Random(self._subject['id']+700).shuffle(neg_emotions)
        if (self._subject['id'] % 4) <= 1:
            self.blocks = [pos_emotions, ['neutral'], neg_emotions]
        else:
            self.blocks = [neg_emotions, ['neutral'], pos_emotions]
        # Get the script matching log and candidate acting scripts
        self._script_dir = os.path.join(
            self._exp_dir, 'stimuli_exp3')
        self._matchFile = os.path.join(self._script_dir, 'script_matching.csv')
        self.script_log = pd.read_csv(self._matchFile,
                                      dtype={e: 'str' for e in self._target_emotions.keys()})
        df1 = self.script_log.copy(deep=True)
        for e in self._target_emotions.keys():
            df1[e] = [len(df1[df1[e] == str(id)]) for id in df1['id']]
        df2 = df1[(df1['gender'] == self._subject['gender'])
                  & (df1['id'] != self._subject['id'])]
        self._candidates = {e: df2[['id', e]]
                            for e in self._target_emotions.keys()}
        # This will be used to record the information of each trial
        self.trial_log = pd.DataFrame({'targetEmotion': pd.Series(dtype='str'),
                                        'actCompletedPerc': pd.Series(dtype='float'),
                                        'act_anger': pd.Series(dtype='float'),
                                        'act_disgust': pd.Series(dtype='float'),
                                        'act_fear': pd.Series(dtype='float'),
                                        'act_sadness': pd.Series(dtype='float'),
                                        'act_amusement': pd.Series(dtype='float'),
                                        'act_inspiration': pd.Series(dtype='float'),
                                        'act_joy': pd.Series(dtype='float'),
                                        'act_tenderness': pd.Series(dtype='float'),
                                        'feel_anger': pd.Series(dtype='float'),
                                        'feel_disgust': pd.Series(dtype='float'),
                                        'feel_fear': pd.Series(dtype='float'),
                                        'feel_sadness': pd.Series(dtype='float'),
                                        'feel_amusement': pd.Series(dtype='float'),
                                        'feel_inspiration': pd.Series(dtype='float'),
                                        'feel_joy': pd.Series(dtype='float'),
                                        'feel_tenderness': pd.Series(dtype='float'),
                                        'self_valence': pd.Series(dtype='float'),
                                        'self_arousal': pd.Series(dtype='float'),
                                        'others_valence': pd.Series(dtype='float'),
                                        'others_arousal': pd.Series(dtype='float'),
                                        'innerDriven': pd.Series(dtype='float'),
                                        'outerDriven': pd.Series(dtype='float'),
                                        'liking': pd.Series(dtype='float'),
                                        'familiarity': pd.Series(dtype='float'),
                                        'actingCredibility': pd.Series(dtype='float'),
                                        'scriptCredibility': pd.Series(dtype='float'),
                                        'emotionCredibility': pd.Series(dtype='float'),
                                        'roleConfidence': pd.Series(dtype='float')})

        # Set the main instructions
        self.welcome_guide = f'''欢迎参加本实验！

接下来我们将为您提供一系列独白剧本，请您
扮演剧本中的主人公、讲述其曾经历过的事件，
它们可能唤起积极、消极或中性情绪。我们会
对您的表演进行录像，但绝不会将其用于学术
研究以外的用途。

按【空格/SPACE】键继续'''
        self.practice_guide = '''为了帮助您更好地熟悉实验流程，接下来请练
习一个完整的表演任务，然后练习一轮数学等
式判断任务。练习过程中如有任何疑问，请随
时向主试问询。

按【空格/SPACE】键继续'''
        self.act_guide = '''接下来请注视屏幕上呈现的十字，想象面前有
几位朋友对您的经历很感兴趣。请您扮演刚才
这个剧本的主人公，讲述您回忆的这件事以及
您的感受。

讲述过程最长为180秒，当您讲述完
请按【退出/ESC】键结束录制。

按【空格/SPACE】键开始'''

    def rehersal_guide(self, emotion: str) -> str:
        """Return rehersal instruction of the given `emotion`."""
        if emotion == 'neutral':
            text = '''《一件对我来说没有什么特别感受的事》

要求1：阅读过程中，如果您发现自己曾经历过
同样的事，请切换一个剧本。

要求2：请您以剧本为参考，在阅读时自由进行
表演前的准备工作，但不必完全复述参考内容。

要求3：在您完成阅读和演前准备工作后，请进行
正式表演。正式表演只录制一次，开始后将无法
中断重来。

按【空格/SPACE】键开始阅读并进行演前准备'''
        else:
            text = f'''《一件曾让我感到非常{self._target_emotions[emotion]}的事》

要求1：阅读过程中，如果您发现自己曾经历过
同样的事，请切换一个剧本。

要求2：请您以剧本为参考，在阅读时自由进行
表演前的准备工作，但不必完全复述参考内容。

要求3：在您完成阅读和演前准备工作后，请进行
正式表演。正式表演只录制一次，开始后将无法
中断重来。

按【空格/SPACE】键开始阅读并进行演前准备'''
        return text

    def _get_script(self, emotion: str, log: bool = True) -> List[str]:
        """Pop a script (formatted by multi-lines in multi-pages) and record its id if `log`."""
        found_one = False
        while not found_one:
            df = self._candidates[emotion]
            assert len(
                df) > 0, '\nMatch failed: all candidate scripts have been traversed!'
            if log:
                script_id = df.loc[df[emotion] == min(
                    df[emotion]), 'id'].sample().iloc[0]
            else:
                script_id = df.loc[df[emotion] == max(
                    df[emotion]), 'id'].sample().iloc[0]
            self._candidates[emotion] = df[df.id != script_id]
            if os.path.exists(os.path.join(self._script_dir, str(script_id))):
                for filename in os.listdir(os.path.join(self._script_dir, str(script_id))):
                    if emotion in filename:
                        with open(os.path.join(self._script_dir, str(script_id), filename), 'r', encoding='utf-8') as f:
                            script = f.read().replace('\n', '').replace(' ', '')
                        found_one = True
        print(
            f'Match to [script {str(script_id)}] with {len(self._candidates[emotion])} candidates remaining.')
        if log:
            self.script_log.loc[self.script_log.id ==
                                self._subject['id'], emotion] = str(script_id)
            self.script_log.to_csv(self._matchFile, index=False,
                                   index_label=False, mode='w')
        n_words_per_line = 30
        n_lines_per_page = 15
        n_words_per_page = n_words_per_line * n_lines_per_page
        script = [script[i:i+n_words_per_page]
                  for i in range(0, len(script), n_words_per_page)]
        script = ['\n'.join(page[i:i+n_words_per_line]
                            for i in range(0, len(page), n_words_per_line)) for page in script]
        return script

    def show_script(self, emotion: str, log: bool = True) -> None:
        """Display the script of the given `emotion` and record it if `log`."""

        # Prepare the visual elements for page control
        title = TextBox2(self._win,
                         font='Microsoft YaHei',
                         letterHeight=0.06,
                         units='norm',
                         alignment='center',
                         pos=(0, 0.8),
                         text='《一件对我来说没有什么特别感受的事》' if emotion == 'neutral' else f'《一件曾让我感到非常{self._target_emotions[emotion]}的事》')
        main_area = TextStim(self._win,
                             font='Microsoft YaHei',
                             #  letterHeight=0.05,
                             height=0.05,
                             units='norm',
                             #  alignment='center',
                             pos=(0, 0.1),
                             #  size=(1.4, 1.4),
                             #  borderColor='white',
                             text='')
        main_box = Rect(self._win,
                        size=(1, 1.2),
                        pos=(0, 0.1),
                        units='norm',
                        lineColor='white')
        change_btn = TextBox2(self._win,
                              text='我经历过\n换个剧本',
                              font='Microsoft YaHei',
                              letterHeight=0.05,
                              color='black',
                              size=[0.2, 0.2],
                              fillColor='white',
                              alignment='center',
                              pos=[-0.75, 0])
        confirm_btn = TextBox2(self._win,
                               text='准备好了\n开始表演',
                               font='Microsoft YaHei',
                               letterHeight=0.05,
                               color='black',
                               size=[0.2, 0.2],
                               fillColor='white',
                               alignment='center',
                               pos=[0.75, 0])
        next_btn, return_btn, _, _, _, _ = self._buttons()
        left_pos, middle_pos, right_pos = [-0.3, -0.6], [0, -0.6], [0.3, -0.6]

        def _draw_page(main_text: str, *other_elements):
            """Draw and show visual elements."""
            title.draw()
            main_box.draw()
            main_area.text = main_text
            main_area.draw()
            change_btn.draw()
            confirm_btn.draw()
            for element in other_elements:
                if isinstance(element, TextBox2):
                    element.draw()
                elif isinstance(element, tuple):
                    assert isinstance(element[0], TextBox2) and isinstance(
                        element[1], list)
                    element[0].pos = element[1]
                    element[0].draw()
            self._win.flip()

        def _show_feedback(feedback: str, duration: float = 1):
            """Gives feedback on keystrokes and stays for a short period of time."""
            _draw_page(feedback)
            core.wait(duration)

        # Choose a script and initialize the page state
        script = self._get_script(emotion, log)
        n_pages = len(script)
        page_flag = 0
        self._mouse.setVisible(True)
        confirmed = False
        while not confirmed:
            # state: one-page script
            if n_pages == 1:
                _draw_page(script[page_flag])
                # action: change a script
                if self._mouse.isPressedIn(change_btn):
                    script = self._get_script(emotion, log)
                    n_pages = len(script)
                    page_flag = 0
                    _show_feedback('...切换剧本中...')
                # action: confirm this script and start to act
                elif self._mouse.isPressedIn(confirm_btn):
                    self._mouse.setVisible(False)
                    confirmed = True
            # state: multi-pages script
            elif n_pages > 1:
                # state: the first page
                if page_flag == 0:
                    _draw_page(script[page_flag], (next_btn, middle_pos))
                    # action: change a script
                    if self._mouse.isPressedIn(change_btn):
                        script = self._get_script(emotion, log)
                        n_pages = len(script)
                        page_flag = 0
                        _show_feedback('...切换剧本中...')
                    # action: confirm this script and start to act
                    elif self._mouse.isPressedIn(confirm_btn):
                        self._mouse.setVisible(False)
                        confirmed = True
                    # action: switch to the next page of this script
                    elif self._mouse.isPressedIn(next_btn):
                        page_flag += 1
                        core.wait(0.5)
                # state: neither the first nor the last page
                elif page_flag < (n_pages-1):
                    _draw_page(script[page_flag], (return_btn,
                               left_pos), (next_btn, right_pos))
                    # action: change a script
                    if self._mouse.isPressedIn(change_btn):
                        script = self._get_script(emotion, log)
                        n_pages = len(script)
                        page_flag = 0
                        _show_feedback('...切换剧本中...')
                    # action: confirm this script and start to act
                    elif self._mouse.isPressedIn(confirm_btn):
                        self._mouse.setVisible(False)
                        confirmed = True
                    # action: switch to the next page of this script
                    elif self._mouse.isPressedIn(next_btn):
                        page_flag += 1
                        core.wait(0.5)
                    # action: switch to the previous page of this script
                    elif self._mouse.isPressedIn(return_btn):
                        page_flag -= 1
                        core.wait(0.5)
                # state: the last page
                elif page_flag == (n_pages-1):
                    _draw_page(script[page_flag], (return_btn, middle_pos))
                    # action: change a script
                    if self._mouse.isPressedIn(change_btn):
                        script = self._get_script(emotion, log)
                        n_pages = len(script)
                        page_flag = 0
                        _show_feedback('...切换剧本中...')
                    # action: confirm this script and start to act
                    elif self._mouse.isPressedIn(confirm_btn):
                        self._mouse.setVisible(False)
                        confirmed = True
                    # action: switch to the previous page of this script
                    elif self._mouse.isPressedIn(return_btn):
                        page_flag -= 1
                        core.wait(0.5)

    def act_task(self,
                 duration: Union[float, int],
                 prepare: Union[float, int, None] = None,
                 log: bool = True,
                 save: bool = True,
                 emotion: Union[str, None] = None,
                 skip: str = 'escape') -> None:
        """Count down for the subject's acting. Meanwhile, record a video (optional).

        Parameters
        ------
        duration : float or int
            Max time limit of acting.
        prepare : float, int, or None, default=None
            If not None, countdown `prepare` time (in seconds) before acting.
        log : bool, default=True
            The flag controlling whether to treat this acting as a formal trial. If True, send trigger and record 'actCompletedPerc' in `self.trials`.
        save : bool, default=True
            The flag controlling whether to record and save a video of this acting.
        emotion : str or None, default=None
            If `save`=True, use `emotion` to name the saved mp4 file.
        skip : str, default='escape'
            A key that when pressed ends the currently acting.

        Note
        ------
        The `save` feature is very demanding on both cpu and memory, and creates a thread block for a certain amount of time after the acting is over (3 minutes of recording -> about 2 minutes of blocking). Therefore, treat the blocking time as a break session.

        See Also
        ------
        Method `_get_camera()`.
        """
        centerMessage = TextBox2(self._win,
                                 font='Microsoft YaHei',
                                 letterHeight=0.15,
                                 alignment='center',
                                 pos=(0, 0),
                                 text='')
        timeReminder = TextBox2(self._win,
                                font='Microsoft YaHei',
                                letterHeight=0.07,
                                alignment='center',
                                pos=(0, 0.8),
                                text=f'剩余{math.ceil(duration)}秒')
        enddingSound = Sound(os.path.join(self._exp_dir, 'dee.mp3'))
        camera = self._get_camera()
        if save:
            assert emotion, "The output recording file must be named with parameter 'emotion'."
        # Count down before task
        if prepare:
            preTimer = core.CountdownTimer(prepare)
            while preTimer.getTime() > 0:
                centerMessage.text = str(math.ceil(preTimer.getTime()))
                centerMessage.draw()
                timeReminder.draw()
                self._win.flip()
        # Task start
        recordTime = duration
        centerMessage.text = '+'
        camera.open()
        camera.record()
        if log:
            self.send_trigger(55)
        while camera.recordingTime < duration:
            camera.update()
            timeReminder.text = f'剩余{math.ceil(duration - camera.recordingTime)}秒'
            timeReminder.draw()
            centerMessage.draw()
            self._win.flip()
            if event.getKeys(keyList=[skip]):
                if log:
                    self.send_trigger(50)
                recordTime = camera.recordingTime
                break
        if log:
            self.send_trigger(56)
            self.trial_log.at[self.trial_idx,
                               'actCompletedPerc'] = 100*(recordTime/duration)
        enddingSound.play()
        camera.stop()
        camera.close()
        centerMessage.text = '表演结束\n\n请记住当前状态\n并放松休息一会\n\n（不要按任何键）'
        centerMessage.draw()
        self._win.flip()
        if save:
            camera.save(os.path.join(
                self._save_dir, f'{self._exp_id}_{self._date}{self._start_time}_{emotion}.mp4'), mergeAudio=True)

    def get_ratings(self,
                    log: bool = True):
        """Get the self-reported ratings about the acting task.

        Parameters
        ------
        log : bool, default=True
            The flag controlling whether to treat this acting as a formal trial. If True, record ratings in `self.trials`.

        Rating Items
        ------
            See `self.trials`.
        """
        # Get elements in different pages
        pages = [self._emotion_ratings(target='刚才您【试图表达】的情绪强度', prefix='act_'),
                 self._emotion_ratings(
                     target='刚才您【实际体验到】的情绪强度', prefix='feel_'),
                 self._4item_ratings(instructions=['总的来说，【您的】主观感受是：',
                                                   None,
                                                   '总的来说，您认为【观众的】主观感受可能是：',
                                                   None],
                                     itemNames=['self_valence',
                                                'self_arousal',
                                                'others_valence',
                                                'others_arousal'],
                                     tickLabels=[('消极', '积极'),
                                                 ('平静', '强烈'),
                                                 ('消极', '积极'),
                                                 ('平静', '强烈')],
                                     startValues=[4, 1, 4, 1]),
                 self._4item_ratings(instructions=['您的表演在多大程度上是由【内在感受】驱动的：',
                                                   '您的表演在多大程度上是由【外在行为】驱动的：',
                                                   '您对【这个剧本】的看法是：',
                                                   None],
                                     itemNames=['innerDriven',
                                                'outerDriven',
                                                'familiarity',
                                                'liking'],
                                     tickLabels=[('完全不', '完全'),
                                                 ('完全不', '完全'),
                                                 ('不熟悉', '熟悉'),
                                                 ('不喜欢', '喜欢')],
                                     startValues=[1, 1, 1, 4]),
                 self._4item_ratings(instructions=['您认为【刚才这段表演】的可信度是：',
                                                   '您认为【这个剧本】的可信度是：',
                                                   '您认为【您表演的情绪】的可信度是：',
                                                   '在表演时，您有多相信您就是这段经历的主人公：'],
                                     itemNames=['actingCredibility',
                                                'scriptCredibility',
                                                'emotionCredibility',
                                                'roleConfidence'],
                                     tickLabels=[('完全不可信', '完全可信'),
                                                 ('完全不可信', '完全可信'),
                                                 ('完全不可信', '完全可信'),
                                                 ('完全不相信', '完全相信')],
                                     startValues=[1, 1, 1, 1])]

        # Draw pages and get ratings
        self._rating_control(pages, log)

        # Overwrite the local file to update ratings
        if log:
            self.trial_log.to_csv(self._ratingFile, index=False,
                                   index_label=False, mode='w')

    def end(self):
        """The routine at the end of the experiment:
            - display the mouse
            - close `self._win` and `self._port`
            - save `self.trials` and `self.math`
            - update `script_matching.csv`"""
        self._end()
        # Final update the local files
        self.trial_log.to_csv(self._ratingFile, index=False,
                               index_label=False, mode='w')
        self.math_log.to_csv(self._mathFile, index=False,
                             index_label=False, mode='w')
        self.script_log.to_csv(self._matchFile, index=False,
                               index_label=False, mode='w')


class EmotionConceptSimilarityTask(Experiment):
    """Individual-level emotion concept similarity (ECS) ratings (PsychoPy; no fNIRS).

    Output file: ``trait_ecs.csv`` in the participant folder (same trait level as
    ``trait_beq.csv`` / ``trait_qcae.csv`` / ``trait_bfi2.csv``).

    Parameters
    ------
    exp_dir : str
        The folder path that contains all nessecary files supporting this experiment.
    stimuli_screen, observe_screen : int
        Indeces of the screens for showing stimuli, and for you to observe computer activities.

    Attributes
    ------
    n_pairs: int
        Total number of emotion-word-pairs.
    trial_idx: int
        Changeable index of the current trial.
    trial_log : pd.DataFrame
        Columns include 'emotion1', 'emotion2', and 'similarity'.
    welcome_guide : str
        The main instruction of this experiment.

    Methods
    ------
    start()
        Create and open an instance of `Window` for presenting experiment stimuli.
    show_instruction(text, wait, pos=(0.,0.)) -> List[str] or None
        Show instruction text in `self._win` and wait for a period.
    judgment()
        Get the ratings about the conceptual similarity of two emotions.
    end()
        End this experiment.
    """

    def __init__(self, exp_dir, stimuli_screen, observe_screen) -> None:
        super().__init__(exp_dir=exp_dir,
                         exp_id='ecs',
                         stimuli_screen=stimuli_screen,
                         observe_screen=observe_screen)
        self._set_subject_info()
        self._set_experiment_info()
        self._ratingFile = os.path.join(self._save_dir, 'trait_ecs.csv')

        # Randomly arrange the positions of 9 emotion words
        self._all_pos = [((-0.4)+x*0.4, 0+y*0.4)
                         for x in range(3) for y in range(3)]
        random.Random(int(datetime.now().strftime('%S'))
                      ).shuffle(self._all_pos)

        # Randomize the sequence of emotion-word-pairs
        self._emotion_pairs = list(
            combinations(self._target_emotions.keys(), 2))
        random.Random(self._subject['id']).shuffle(self._emotion_pairs)
        self.n_pairs = len(self._emotion_pairs)

        # This will be used to record the similarity rating of each pairs
        self.trial_log = pd.DataFrame({'emotion1': pd.Series(dtype='str'),
                                    'emotion2': pd.Series(dtype='str'),
                                    'similarity': pd.Series(dtype='int')})

        # Set the main instructions
        self.welcome_guide = f'''欢迎参加本实验！

接下来，您需要对九种情绪概念之间的相似度
进行评估。这九种情绪的名称将以文字的形式
同时呈现在屏幕上，不过每一次判断您只需要
考虑【有白色边框】的两种情绪。请使用鼠标
拖动下方的滑动条来进行操作，操作完成后按
【空格/SPACE】进行提交。

按【空格/SPACE】键继续'''

    def judgment(self, submit_key='space'):
        """Rating the conceptual similarity of two emotions."""
        self.trial_idx += 1
        targets = self._emotion_pairs[self.trial_idx]
        self.trial_log.at[self.trial_idx, 'emotion1'] = targets[0]
        self.trial_log.at[self.trial_idx, 'emotion2'] = targets[1]

        # Prepare the visual elements
        words = []
        for i, (e_eng, e_chn) in enumerate(self._target_emotions.items()):
            word = TextBox2(self._win,
                            text=e_chn,
                            font='Microsoft YaHei',
                            letterHeight=0.06,
                            units='norm',
                            alignment='center',
                            size=(0.3, 0.3),
                            borderColor='white' if e_eng in targets else 'black',
                            pos=self._all_pos[i])
            words.append(word)
        labels = []
        for i in range(2):
            label = TextBox2(self._win,
                             text='非常不相似' if i == 0 else '非常相似',
                             font='Microsoft YaHei',
                             letterHeight=0.06,
                             units='norm',
                             alignment='center',
                             pos=(-0.6+i*1.2, -0.6))
            labels.append(label)
        rating_guide = TextBox2(self._win,
                                text='请拖动滑动条来判断以上两种情绪概念的相似程度',
                                font='Microsoft YaHei',
                                letterHeight=0.06,
                                units='norm',
                                alignment='center',
                                pos=(0, -0.42))
        submit_guide = TextBox2(self._win,
                                text='按【空格/SPACE】键提交',
                                font='Microsoft YaHei',
                                letterHeight=0.06,
                                units='norm',
                                alignment='center',
                                pos=(0, -0.78))
        slider = Slider(self._win,
                        markerColor='White',
                        ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                        pos=(0, -0.6),
                        size=(1, 0.03))
        slider.reset()

        # Record the rating
        self._mouse.setVisible(True)
        confirmed = False
        while not confirmed:
            for word in words:
                word.draw()
            for label in labels:
                label.draw()
            rating_guide.draw()
            submit_guide.draw()
            slider.draw()
            self._win.flip()
            if slider.getRating():
                self.trial_log.at[self.trial_idx,
                               'similarity'] = slider.getRating()
                if event.getKeys(submit_key):
                    confirmed = True
        self._mouse.setVisible(False)

        # Overwrite the local file to update ratings
        self.trial_log.to_csv(self._ratingFile, index=False,
                           index_label=False, mode='w')

    def end(self):
        """The routine at the end of the experiment:
            - display the mouse
            - close `self._win`
            - save `self.trials`"""
        self._end()
        self.trial_log.to_csv(self._ratingFile, index=False,
                           index_label=False, mode='w')
