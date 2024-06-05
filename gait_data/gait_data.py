import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from gait_data import gait_utils
from scipy import signal
from scipy.fftpack import fft

class GaitData:

    def __init__(self, video_name, keypoints_csv, joints, cow_id=-1, use_kp=None, smoothing='savgol', smoothing_params=None):
        self.video_name = video_name
        self.keypoints_csv = keypoints_csv
        self.joints = joints
        self.cow_id = cow_id
        self.use_kp = use_kp  ## If None, use all the keypoints OR Use a selection of keypoints only
        self.smoothing = smoothing
        self.smoothing_params = smoothing_params

        self.keypoints = None
        self.n_kp = 0
        self.keypoints_header = None
        self.keypoints_meta_data = None
        self.smoothed_keypoints = None
        self.keypoints_meta_data = None
        self.n_frames = 0
        self.steps = None

        self.set_keypoints(keypoints_csv)
        self._set_kp_names_idx()
        self.smooth_keypoints(smoothing, smoothing_params)
        self._trim_keypoints(5,-5)
        self.trim_first_last_step()

    def set_keypoints(self, keypoints_csv):
        """
        Reads the keypoint csv file and sets it as a class variable.
        :param keypoints_csv: The path to the csv file
        """
        header = np.genfromtxt(keypoints_csv, dtype=str, delimiter=',', max_rows=1)
        self.n_kp = (header.shape[0] - 2) // 3

        if self.use_kp is not None:
            usecols = [0,1]
            for kp_name in self.use_kp:
                for i in range(2, header.shape[0]):
                    if (kp_name in header[i]) and (not 'likelihood' in header[i]):
                        usecols.append(i)
        else:
            usecols = self._get_kp_cols()

        kp = np.genfromtxt(keypoints_csv, dtype=str, delimiter=',', usecols=usecols)
        self.n_kp = (len(usecols) - 2) // 2
        self.keypoints = kp[1:, 2:].astype(float)
        self.keypoints_header = kp[0]
        self.keypoints_meta_data = kp[1:, :2]
        self.n_frames = self.keypoints.shape[0]

    def _get_kp_cols(self):
        """
        Generates the column coordinates to be read from the keypoint csv file.
        Columns to be read:
          * 0: video_name
          * 1: frame
          * 2: x coordinate of 1st keypoint
          * 3: y coordinate of 1st keypoint
          * [4]: likelihood of 1st keypoint. SKIP THIS ONE
          * 5: x coordinate of 2nd keypoint
          * 6: y coordinate of 2st keypoint
          * [7]: likelihood of 2nd keypoint. SKIP THIS ONE
          * ... etc. (Starting from idx 2, we only take x and y columns, and discard the likelihood column.
        :return: list of column indices
        """
        usecols = [0, 1]  # video_name, frames
        for i in range(self.n_kp):  # x and y coordinates, ignore_likelihood
            x_col = 2 + (i * 3)
            usecols.extend([x_col, x_col + 1])
        return usecols

    def _get_kp_names_idx(self, overwrite=False):
        """
        Gets the name of the keypoints and their x,y index
        :return: Dictionary with name of keypoint as a key, and tuple of x,y coordinates as value
        """
        if (self.kp_names is None) or overwrite:
            self._set_kp_names_idx()
        return self.kp_names

    def _set_kp_names_idx(self):
        kp_names = {}
        for i in range(2, self.keypoints_header.shape[0], 2):
            kp_name = self.keypoints_header[i].split("_")[0]
            kp_names[kp_name] = (i-2, i-1)

        self.kp_names = kp_names

    def get_xy_bodyparts(self, bodyparts, smoothed=False):
        """
        Returns the x,y keypoints of the specified bodyparts
        :param bodyparts: List of keypoint names, e.g. ["LFHoof", "LFAnkle"]
        :return: array with the x,y keypoints of the bodyparts. Shape: (n_frames, len(bodyparts),  2)
        """
        if bodyparts == "all":
            bodyparts = self.kp_names.keys()
        xys = []

        for i, kp in enumerate(bodyparts):
            xys.extend(self.kp_names[kp])

        keypoints = self.smoothed_keypoints if smoothed else self.keypoints
        data = np.take(keypoints, xys, axis=1)
        # transform the dataframe into an array of size (n_frames, n_keypoints, 2)
        data = data.reshape((data.shape[0], -1, 2))

        return data

    def smooth_keypoints(self, method='savgol', smooth_params=None):
        """
        Smoothes the keypoints using the specified smoothing method.
        :param method: Smoothing method to use. Default is 'savgol', for Savitzky-Golay filter
        :param smooth_params: List of parameters to use for smoothing. Default is None.
        """
        if self.n_frames <= 3:
            self.smoothed_keypoints = self.keypoints
            print("Warning: number of frames is too small for smoothing.")
            return

        supported_methods = ["savgol", "median", "gaussian", "hampel", "outliers", 'no']
        if method == "savgol":
            if smooth_params is None:
                self.smoothed_keypoints = gait_utils.savgol_smoothing(self.keypoints)
            else:
                keypoints = np.copy(self.keypoints)
                keypoints = keypoints.reshape((self.keypoints.shape[0], -1, 2))
                for i in range(len(smooth_params)):  # x,y
                    keypoints[:, 0:14, 0] = gait_utils.savgol_smoothing(self.keypoints[:, 0:14, 0], window_length=smooth_params[0]["window_length"], polyorder=smooth_params[0]["polyorder"])
                    keypoints[:, 14:, 0] = gait_utils.savgol_smoothing(self.keypoints[:, 14:, 0], window_length=smooth_params[0]["window_length"]*2, polyorder=smooth_params[0]["polyorder"])
                self.smoothed_keypoints = keypoints

        elif method == "median":
            self.smoothed_keypoints = gait_utils.median_smoothing(self.keypoints)
        elif method == "gaussian":
            self.smoothed_keypoints = gait_utils.gaussian_smoothing(self.keypoints)
        elif method == "hampel":
            # self.smoothed_keypoints, _ = gait_utils.hampel_filter(self.keypoints, 6, 3, replace=True)
            self.smoothed_keypoints = gait_utils.rolling_hampel(self.keypoints, 50, 5)
        elif method == "outliers":
            self.keypoints = gait_utils.filter_keypoints(self.keypoints, trim=None, smooth=False)
            self.smooth_keypoints("savgol", smooth_params=smooth_params)
        #
        elif method == 'no':
            self.smoothed_keypoints = np.copy(self.keypoints)
        else:
            raise ValueError(" ".join(["Agrument 'method' must have one of the following values:", *supported_methods]))

    def _trim_keypoints(self, start, end):
        self.keypoints = self.keypoints[start:end, :]
        self.keypoints_meta_data = self.keypoints_meta_data[start:end, :]
        self.n_frames = self.keypoints.shape[0]

        if self.smoothed_keypoints is not None:
            self.smoothed_keypoints = self.smoothed_keypoints[start:end, :]

        self.get_steps(overwrite=True)

    def trim_trajectory(self, start=1, end=-2):
        """
        Trim the start and end of the kp trajectories to the first complete step, and last complete step.
        By default, we discard at the first start_contact, and the last end_contact.
        :param start:
        :param end:
        :param steps:
        :return:
        """
        assert start >= 0 and start <= 3, "start should be a value in range [0,3]"
        assert end <= -1 and end >= -4, "end should be a value in range [-4,-1]"

        steps = self.steps
        if steps is None:
            # Here we don't set the self.steps because we will be trimming the keypoints,
            # so these steps won't be valid after the trimming.
            steps = self.find_steps()

        start_contacts = []
        end_contacts = []
        legs = []

        for i, leg in enumerate(steps.keys()):
            start_contact = steps[leg]['start_contact']
            end_contact = steps[leg]['end_contact']
            start_contacts.append(start_contact[0])
            end_contacts.append(end_contact[-1])
            legs.append(leg)

        start_contacts = np.sort(np.array(start_contacts))
        start_trim = start_contacts[start]
        end_contacts = np.sort(np.array(end_contacts))
        end_trim = end_contacts[end]

        self._trim_keypoints(start_trim, end_trim)

    def trim_first_last_step(self):
        steps = self.steps
        if steps is None:
            # Here we don't set the self.steps because we will be trimming the keypoints,
            # so these steps won't be valid after the trimming.
            steps = self.find_steps()

        start_contacts = []
        end_contacts = []
        legs = []

        for i, leg in enumerate(steps.keys()):
            start_contact = steps[leg]['start_contact']
            end_contact = steps[leg]['end_contact']
            start_contacts.append(start_contact[0])
            end_contacts.append(end_contact[-1])
            legs.append(leg)

        start_contacts = np.sort(np.array(start_contacts))
        start_trim = start_contacts[-1]  # latest first start contact
        end_contacts = np.sort(np.array(end_contacts))
        end_trim = end_contacts[0]  # earliest last contact

        self._trim_keypoints(start_trim, end_trim)

    def get_steps(self, overwrite=False):
        if (self.steps is None) or overwrite:
            self.steps = self.find_steps()

        return self.steps

    def find_steps(self, legs="all", step_prefix="Hoof"):
        """
        Find the steps for each leg based on the hoof location
        :param legs: Legs to find the steps from. Default: "all"
        :param step_prefix: Name of the keypoint to find the steps from. Default: "Hoof".
        :return: Dict of steps for each leg.
        """
        # compute steps
        if legs == 'all':
            legs = ['LF', 'LH', 'RF', 'RH']

        steps = {}
        for fh in ['F', 'H']:
            lr_x_signals = []
            lr_legs = []
            for lr in ['L', 'R']:
                leg = lr+fh
                bodypart = leg + step_prefix
                kp_bodypart = self.get_xy_bodyparts([bodypart], smoothed=True)
                x_signal, y_signal = kp_bodypart[:, 0].T

                # Find start and end of hoof on the floor. ==> anything in between means the leg is in stance phase
                # plateaus, dF, d2F = gait_utils.find_plateaus(x_signal, tolerance=0.75, min_length=10, smoothing=3)
                plateaus = gait_utils.find_plateaus_numpy(x_signal, tolerance=10, min_length=10, smoothing=5)
                start_contact, end_contact = plateaus.T

                # start_contact, end_contact = gait_utils.find_plateaus_2(x_signal, 4, 40)

                # Find midway of the step (when the step is mid-air (ish).
                # Here I'm looking at the second derivative of the x signal.
                # I think the second derivative represents the acceleration along the x-axis?
                # So, the peak of the acceleration is half-way through the step.
                # Could also simply look at y-signal, a peak is where the hoof is mid-air. However, the signal is veryyyy noisy
                # and it didn't work very well. Will stick with the d2F of the x signal for now.
                grad = np.gradient(x_signal, edge_order=2)
                peaks, properties = signal.find_peaks(grad, height=20, width=0, prominence=2, plateau_size=[None, None])
                # peaks, properties = signal.find_peaks(dF, height=20, width=0, prominence=2, plateau_size=[None, None])

                steps[leg] = {'bodypart': bodypart,
                              'peaks': peaks,
                              'start_contact': start_contact,
                              'end_contact': end_contact,
                              'x_signal': x_signal}
                lr_x_signals.append(x_signal)
                lr_legs.append(leg)

            intersect_0 = np.argwhere(np.diff(np.sign(np.array(lr_x_signals[0]) - np.array(lr_x_signals[1])))).flatten()
            intersect_1 = np.argwhere(np.diff(np.sign(np.array(lr_x_signals[1]) - np.array(lr_x_signals[0])))).flatten()
            steps[lr_legs[0]]['intersect'] = intersect_1
            steps[lr_legs[1]]['intersect'] = intersect_0

        return steps

    def min_max_steps(self):
        steps = self.get_steps()
        min_steps = np.inf
        max_steps = 0
        for leg in steps.keys():
            n_steps = steps[leg]['start_contact'].shape[0]
            min_steps = np.min([min_steps, n_steps])
            max_steps = np.max([max_steps, n_steps])

        return min_steps, max_steps

    def gait_cycle_length(self):
        steps = self.get_steps()
        n_frames_per_leg = []
        for leg in steps.keys():
            n_frames_per_leg.extend(np.diff(steps[leg]['start_contact']))

        return np.max(n_frames_per_leg)


    def head_bobbing(self):
        """
        Average difference in head-height between the front legs and hind legs.
        :return: a dict with the height-diff for the front and hind leg.
        """
        self.get_steps()  # sets the steps if they are None

        distances = {'H': [], 'F': []}

        # First Hind legs, then Front legs
        # Keypoint coordinates of the Nose
        kp_nose = self.get_xy_bodyparts(["Nose"], smoothed=True)
        x, y = kp_nose[:, 0].T

        for fh in ['H', 'F']:
            l_start = self.steps["L" + fh]['start_contact']  # contact frames of left leg
            r_start = self.steps["R" + fh]['start_contact']  # contact frames of right leg

            i_range = min(l_start.shape[0], r_start.shape[0])  # in case there is more steps on one side
            dists = y[l_start[:i_range]] - y[r_start[:i_range]]
            distances[fh] = np.mean(dists)

        return distances

    def head_amplitude(self):
        """
        Head bobbing amplitude using Fast Fourier Transforms
        """
        self.get_steps()
        kp_head = self.get_xy_bodyparts(["HeadTop"], smoothed=True)

        x, y = kp_head[:, 0].T

        fft_y = fft(y)  # Fast Fourrier Transform of the y-signal
        N = len(fft_y)  # Number of frames
        n = np.arange(N)
        sr = 30  # sampling rate: 30 frames per second: 30/1 Hz
        T = N / sr  #
        freq = n / T

        n_oneside = N // 2  # Get the one-sided specturm

        t_f = sr / (freq[1:n_oneside] + np.spacing(1))  # Time in frames

        amp = np.abs(fft_y[:n_oneside] / n_oneside)  # amplitude of the signal
        max_amplitude = np.max(amp)  # maximum amplitude

        # gait_cyle_lenght = self.gait_cycle_length()
        # t_f_idx = np.where(t_f <= gait_cyle_lenght)  # frequencies where the time in frames is <= then a gait cycle
        # max_amplitude = np.max(amp[t_f_idx])  # maximum amplitude within a gait cycle

        # max_amplitude = max_amplitude / self.get_head_size(mean=True)
        # t_f_idx_2 = np.where( t_f <= gait_cyle_lenght//2)
        # max_amplitude_half =  np.max(amp[t_f_idx_2]) # maximum amplitude within 1/2 gait cycle
        # return max_amplitude_full, max_amplitude_half

        return max_amplitude

    def tracking_distance(self):
        """
        Calculates the mean tracking distance for left and right legs.
        The tracking distance is the x-distance between the front hoof ground contact and hind hoof ground conctact.
        Ideally, the hind hoof would land in front or exactly where the front hoof was, so dist <= 0.

        Returns
        -------
        A dictionary of the tracking distances per side (left - right).
        """
        self.get_steps()  # sets the steps if they are None

        distances = {}
        for side in ['L', 'R']:
            h_bodypart = self.steps[side + 'H']['bodypart']
            h_start = self.steps[side + 'H']['start_contact']
            h_end = self.steps[side + 'H']['end_contact']
            h_mid = (h_end + h_start) // 2  # frame in the middle of the contact time
            # h_mid = self.steps[side + 'H']['intersect']  # take intersection frame instead of step

            f_bodypart = self.steps[side + 'F']['bodypart']
            f_start = self.steps[side + 'F']['start_contact']
            f_end = self.steps[side + 'F']['end_contact']
            f_mid = (f_end + f_start) // 2  # frame in the middle of the contact time
            # f_mid = self.steps[side + 'F']['intersect']  # frame in the middle of the contact time

            kp_hoofs = self.get_xy_bodyparts([h_bodypart, f_bodypart], smoothed=True)

            h_x, h_y = kp_hoofs[:, 0].T
            f_x, f_y = kp_hoofs[:, 1].T

            # Frame index of hind leg must be larger than frame index front leg, because hind lands where front was.
            # If the index of hind leg < index of front leg, need to shift the comparisons by 1.
            if h_mid[0] <= f_mid[0]:
                h_mid = h_mid[1:]  # shift one to the left

            # remove last to match length
            if f_mid.shape[0] > h_mid.shape[0]:
                f_mid = f_mid[0:h_mid.shape[0]]
            elif f_mid.shape[0] < h_mid.shape[0]:
                h_mid = h_mid[0:f_mid.shape[0]]

            dists = f_x[f_mid] - h_x[h_mid]
            distances[side] = np.median(dists) / self.get_head_size(mean=True)
            # distances[side] = np.median(dists)

        return distances

    def stride_length(self, mean=True):
        """
        Calculates the stride length per leg, that is, the distance between 2 successive contact of the same foot.
        :return: A dictionary of the stride length per leg.
        """
        self.get_steps()  # sets the steps if they are None
        stride_lengths = {'LH':0, 'RH':0, 'LF':0, 'RF':0}

        for k in stride_lengths.keys():
            bodypart = self.steps[k]['bodypart']
            start = self.steps[k]['start_contact']
            end = self.steps[k]['end_contact']
            mid = (end + start) // 2  # frame in the middle of the

            kp_hoofs = self.get_xy_bodyparts([bodypart], smoothed=True)
            x, y = kp_hoofs[:, 0].T

            dists = np.diff(x[mid])

            if mean:
                dists = np.median(dists)

            stride_lengths[k] = dists / self.get_head_size(mean=True)


        return stride_lengths

    def stride_length2(self, mean=True):
        self.get_steps()  # sets the steps if they are None
        stride_ratios = {'F': 0, 'H': 0}
        stride_length = self.stride_length(mean)
        for k in stride_ratios.keys():

            stl_l = stride_length['L'+k]
            stl_r = stride_length['R'+k]

            ratio = abs(stl_l - stl_r)
            stride_ratios[k] = ratio / self.get_head_size(mean=True)

        return stride_ratios

    def stance_duration_difference(self, fps=1):
        self.get_steps()

        stance_diff = {'F': 0, 'H':0}

        for k in stance_diff.keys():
            r_steps = self.steps['R'+k]
            l_steps = self.steps['L'+k]

            r_durations = np.median(r_steps['end_contact'] - r_steps['start_contact'])
            l_durations = np.median(l_steps['end_contact'] - l_steps['start_contact'])

            stance_diff[k] = abs(r_durations - l_durations) / fps  # in seconds

        return stance_diff

    def swing_duration_difference(self, fps=1):
        self.get_steps()

        stride_diff = {'F': 0, 'H': 0}

        for k in stride_diff.keys():
            r_steps = self.steps['R' + k]
            l_steps = self.steps['L' + k]

            r_durations = np.median(r_steps['start_contact'][1:] - r_steps['end_contact'][:-1])
            l_durations = np.median(l_steps['start_contact'][1:] - l_steps['end_contact'][:-1])

            stride_diff[k] = abs(r_durations - l_durations) / fps  # in seconds

        return stride_diff

    def get_head_size(self, smoothed=True, mean=False, bodyparts=None):
        """
        Calculates the head size of the cow, per frame.
        The head size is the euclidian distance between the head-top and the nose keypoints

        Parameters
        ----------
        smoothed: whether to use the smoothed keypoints
        bodyparts: body parts to use to compute the head size. If None, it will use the joints defined in GaitData

        Returns
        -------
        The head size per frame
        """
        if bodyparts is None:
            bodyparts = self.joints['Head']

        keypoints = self.get_xy_bodyparts(bodyparts, smoothed=smoothed)
        head_size = np.linalg.norm(keypoints[:, 0, :] - keypoints[:, 1, :], ord=2, axis=1, keepdims=True)

        if mean:
            return np.median(head_size)

        return head_size

    def compute_back_curvature(self, bodyparts=None, method="BPM", smoothed=True):
        """
        Compute the back curvature of the cow
        :param self: the gait data
        :param bodyparts: list body parts to use for the back curvature. Default: ["Spine1", "Spine3", "Spine2"]
        :param method: Method for the curvature. Default: BPM (Poursaberry 2010)
        :param smoothed: Whether to use to smoothed keypoints for the curvature. Default: True
        :return: the BPM per frame.
        """
        if bodyparts is None:
            bodyparts = self.joints["Spine3"]

        keypoints = self.get_xy_bodyparts(bodyparts, smoothed=smoothed)
        head_size = np.median(self.get_head_size(smoothed=smoothed))
        self.get_steps()  # sets the steps if they are None

        if method == "BPM":
            # 4 legs, 1 BPM per leg.
            # For each leg
            ## Take a frame where the food is mid-step (in the air). The other feet should be on the ground.
            ## We choose a step in the middle of the trajectories
            bpms = []
            for i, leg in enumerate(self.steps.keys()):
                peaks = self.steps[leg]['peaks']
                frame = keypoints[peaks, :]
                bpms.append(np.min(gait_utils.compute_BPM(frame, head_size)))  #keep
            bpms = np.array(bpms)
            return np.median(bpms)  # keep

        elif method== "DEG":
            angles = np.zeros(4)
            for i, leg in enumerate(self.steps.keys()):
                peaks = self.steps[leg]['peaks']
                middle = peaks.shape[0] // 2  # Step in the middle, so if 3 steps for that leg, take step N.2
                middle_frame = peaks[middle]
                frame = keypoints[middle_frame, :]
                angles[i] = gait_utils.compute_back_angle(frame)
            return np.median(angles)
        else:
            print("Method %s is not supported in compute_back_curvature()" % method)
            return None

    ##########################
    ### PLOTTING FUNCTIONS ###
    ##########################
    def plot_xy_trajectories(self, smoothed=False, bodyparts="all", colormap=None, image_path=None,
                                   img_res=(1080, 1920), show_legend=False, show=True, xlim=None, ylim=None, peaks=None):
        if colormap is None:
            # colormap = ['navy', 'mediumblue', 'blue',
            #             'dodgerblue', 'lightskyblue', 'deepskyblue',
            #             'turquoise', 'aquamarine', 'palegreen',
            #             'khaki', 'yellow', 'gold',
            #             'orange', 'darkorange',
            #             'orangered', 'red', 'darkred'
            #             ]
            colormap = ['navy', 'green', 'red',
                        'orange', 'lightskyblue', 'deepskyblue',
                        'red', 'palegreen', 'orange'
                        ]

        fig = plt.figure(figsize=(16, 9))

        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        # if xlim != None:
        #     ax.set_xlim(xlim)
        # if ylim != None:
        #     ax.set_ylim(ylim)

        if bodyparts == "all":
            bodyparts = list(self.kp_names.keys())

        data = self.get_xy_bodyparts(bodyparts, smoothed=smoothed)
        # trick to always get the same plot dimensions
        ax1.set_xlim((0, data.shape[0]))
        ax2.set_xlim((0, data.shape[0]))
        # ax.set_ylim((0, img_res[0]))
        ax1.set_ylim((0, 1920))
        ax2.set_ylim((400, 900))

        # plot one line for each body part
        for bp in range(data.shape[1]):
            x, y = data[:, bp].T
            frames = np.arange(0, x.shape[0])
            ax1.plot(frames, x, c=colormap[bp], label=bodyparts[bp])
            ax2.plot(frames, y, c=colormap[bp], label=bodyparts[bp])

        ax2.invert_yaxis()

        if image_path is not None:
            image = io.imread(image_path)
            image = resize(image, img_res)
            plt.imshow(image)

        if show_legend:
            ax1.legend()
            ax2.legend()

        if show:
            plt.show()

        return fig



    def plot_keypoint_trajectories(self, smoothed=False, bodyparts="all", colormap=None, image_path=None,
                                   img_res=(1080, 1920), show_legend=False, show=True, xlim=None, ylim=None, peaks=None):
        """
        Plots the keypoint trajectories on a graph. It can plot all the body parts or a selection of the bodyparts.
        You can also overlay the last frame of the video.
        :param smoothed: whether to plot the original keypoints or the smoothed ones.
        :param bodyparts: List of the body parts to plot.
                          To plot all body parts, use ["all"], otherwise list them by name, e.g. ["Head", "Nose"]
        :param colormap: List of the colors of the bodyparts.
                         If None, will use the default one defined in this function.
        :param image_path: Path to the image overlay. If None, no image is shown.
        :param img_res: resolution of the video frames. Default is (1080, 1920)
        :param show_legend: Whether to show the plot legend
        :param show: Whether to show the plot. If True, the function will call plt.show()
        :param xlim: limits of the x-axis
        :param ylim: limits of the y-axis
        :return: The figure with the keypoint trajectories.
        """
        if colormap is None:
            # colormap = ['navy', 'mediumblue', 'blue',
            #             'dodgerblue', 'lightskyblue', 'deepskyblue',
            #             'turquoise', 'aquamarine', 'palegreen',
            #             'khaki', 'yellow', 'gold',
            #             'orange', 'darkorange',
            #             'orangered', 'red', 'darkred'
            #             ]
            colormap = ['navy', 'lightskyblue',
                        'turquoise', 'gold',
                        'orange', 'darkorange',
                        'orangered', 'red', 'darkred'
                        ]

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot()

        # trick to always get the same plot dimensions
        ax.set_xlim((0, img_res[1]))
        # ax.set_ylim((0, img_res[0]))
        ax.set_ylim((400, 900))

        if xlim != None:
            ax.set_xlim(xlim)
        if ylim != None:
            ax.set_ylim(ylim)

        if bodyparts == "all":
            bodyparts = list(self.kp_names.keys())

        data = self.get_xy_bodyparts(bodyparts, smoothed=smoothed)
        # plot one line for each body part
        for bp in range(data.shape[1]):
            x, y = data[:, bp].T
            ax.plot(x, y, c=colormap[bp], label=bodyparts[bp])

            if peaks != None:
                ax.plot(x[peaks[bp]], y[peaks[bp]], 'x', c='r')

        ax.invert_yaxis()

        if image_path is not None:
            image = io.imread(image_path)
            image = resize(image, img_res)
            plt.imshow(image)

        if show_legend:
            ax.legend()

        if show:
            plt.show()

        return fig

    def plot_steps(self, legs="all", show=True, xy_lim=(1920, 1080)):
        self.get_steps()  # sets the steps if they are None

        if legs == "all":
            legs = list(self.steps.keys())

        colormap = ['mediumblue', 'lightskyblue', 'orange', 'red']

        # plot the keypoints
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot()

        # trick to always get the same plot dimensions
        ax.set_xlim((0, xy_lim[0]))
        # ax.set_ylim((0, xy_lim[1]))
        ax.set_ylim((700, 900))

        for i, leg in enumerate(legs):
            bodypart = self.steps[leg]['bodypart']
            peaks = self.steps[leg]['peaks']
            start_contact = self.steps[leg]['start_contact']

            kp_bodypart = self.get_xy_bodyparts([bodypart], smoothed=True)

            # plot one line for each body part
            x, y = kp_bodypart[:, 0].T
            ax.plot(x, y, '.', c=colormap[i], label=leg)
            ax.plot(x[start_contact], y[start_contact], 'x', c='r')
            ax.plot(x[peaks], y[peaks], 'x', c='g')

        ax.invert_yaxis()
        ax.legend()

        if show:
            plt.show()
        else:
            return fig

    def plot_x_steps(self, legs="all", show=True, xy_lim=(1920, 1080)):
        self.get_steps()  # sets the steps if they are None

        if legs == "all":
            legs = list(self.steps.keys())

        colormap = ['blue', 'orange', 'green', 'red']

        # plot the keypoints
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot()

        # trick to always get the same plot dimensions

        for i, leg in enumerate(legs):
            bodypart = self.steps[leg]['bodypart']
            peaks = self.steps[leg]['peaks']
            start_contact = self.steps[leg]['start_contact']
            end_contact = self.steps[leg]['end_contact']

            kp_bodypart = self.get_xy_bodyparts([bodypart], smoothed=True)

            # plot one line for each body part
            x, y = kp_bodypart[:, 0].T

            ax.plot(np.arange(x.shape[0]), x, label=leg, color=colormap[i])
            ax.plot(start_contact, x[start_contact], 'og')
            ax.plot(end_contact, x[end_contact], 'or')

            ax.plot(peaks, x[peaks], "x", color=colormap[i])


        ax.legend()
        ax.set_title(self.video_name)

        if show:
            plt.show()
        else:
            return fig

    def to_csv(self, csv_path, smoothed=True):

        if smoothed:
            s_kp = self.smoothed_keypoints
        else:
            s_kp = self.keypoints

        n_cols = (self.n_kp * 3) + 2  # x,y,likelihood * keypoints + video + frame
        kp_df = np.ndarray((self.n_frames + 1, n_cols), dtype='U18')

        kp_df[0, 0:2] = self.keypoints_header[0:2]  # video, frame column name
        kp_df[1:, :2] = self.keypoints_meta_data  # video, frame data

        for i in range(self.n_kp):
            x_col_df = (i * 3) + 2
            x_col = i * 2

            # Column names
            kp_df[0, x_col_df] = self.keypoints_header[x_col+2] # x header
            kp_df[0, x_col_df + 1] = self.keypoints_header[x_col + 3] # y header
            kp_df[0, x_col_df + 2] = self.keypoints_header[x_col+2].split('_')[0]+'_likelihood' # likelihood header

            # keypoint data
            kp_df[1:, x_col_df] = s_kp[:, x_col]  # x
            kp_df[1:, x_col_df + 1] = s_kp[:, x_col + 1]  # y
            kp_df[1:, x_col_df + 2] = '1'  # likelihood


        np.savetxt(os.path.join(csv_path, f'{self.video_name}.csv'), kp_df, fmt='%s', delimiter=',')