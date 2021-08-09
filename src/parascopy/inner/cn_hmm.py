from scipy.stats import nbinom
from scipy.special import logsumexp
from scipy import optimize
from scipy import signal
import numpy as np
import sys
import collections
import operator

from . import common
from .genome import Interval
from . import cn_tools


class HmmModel:
    """
    Class that stores an HMM model.
    Matrices have the following dimensions:
        initial    (n_hidden),
        transition (n_hidden x n_hidden x n_observations-1),
        emission_matrices (n_samples x n_hidden x n_observations).
    max_state_dist defines maximal jump between hidden states:
        abs(j - i) <= max_state_dist for consecutive states i and j.
    All numbers are stored as natural logs.
    """

    def __init__(self, n_samples, n_hidden, n_observations, max_state_dist):
        self._n_samples = n_samples
        self._n_hidden = n_hidden
        self._n_observations = n_observations
        self._max_state_dist = max_state_dist

        self._initial = None
        self._transition = None
        self._emission_matrices = None

        # Vector of natural log probabilities (n_samples).
        self._total_probs = None
        # Three matrices of size (n_samples x n_states x n_observations).
        self._alphas = None
        self._betas = None
        self._gammas = None

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def n_observations(self):
        return self._n_observations

    @property
    def initial(self):
        return self._initial

    @property
    def transition(self):
        return self._transition

    @property
    def emission_matrices(self):
        return self._emission_matrices

    @property
    def max_state_dist(self):
        return self._max_state_dist

    @property
    def total_probs(self):
        return self._total_probs

    @property
    def alphas(self):
        return self._alphas

    @property
    def betas(self):
        return self._betas

    @property
    def gammas(self):
        return self._gammas

    def set_uniform_initial(self):
        self._initial = np.full(self._n_hidden, -np.log(self._n_hidden))

    def set_initial(self, initial):
        if initial.shape != (self._n_hidden,):
            raise ValueError('Cannot set initial probabilities: got shape {}, expected ({},)'
                .format(initial.shape, self._n_hidden))
        self._initial = initial

    def set_transition(self, transition):
        if (self._n_hidden, self._n_hidden) == transition.shape:
            self._transition = np.repeat(transition[:, :, np.newaxis], self._n_observations - 1, axis=2)
        elif (self._n_hidden, self._n_hidden, self._n_observations - 1) == transition.shape:
            self._transition = transition
        else:
            raise ValueError('Cannot set transitions: got shape {shape}, expected {hid},{hid} or {hid},{hid},{obs}'
                .format(shape=transition.shape, hid=self._n_hidden, obs=self._n_observations - 1))

    def set_emission_matrices(self, emission_matrices):
        exp_shape = (self._n_samples, self._n_hidden, self._n_observations)
        if emission_matrices.shape != exp_shape:
            raise ValueError('Cannot set emission probabilities: got shape {}, expected {}'
                .format(emission_matrices.shape, exp_shape))
        self._emission_matrices = emission_matrices

    def consecutive_state_range(self, state):
        return (max(state - self._max_state_dist, 0), min(state + self._max_state_dist + 1, self._n_hidden))

    def viterbi(self, sample_id):
        """
        Runs Viterbi algorithm for a single sample.
        """
        prev_matrix = np.zeros((self._n_hidden, self._n_observations), dtype=np.int16)
        # Two consecutive columns of the values matrix (no need to store more).

        col_a = None
        col_b = self._initial + self._emission_matrices[sample_id, :, 0]
        for obs_ix in range(1, self._n_observations):
            col_a = col_b
            col_b = np.full(self._n_hidden, -np.inf)

            for state, emission in enumerate(self._emission_matrices[sample_id, :, obs_ix]):
                if not np.isfinite(emission):
                    continue
                best_state = state
                best_val = -np.inf

                for prev in range(*self.consecutive_state_range(state)):
                    value = col_a[prev] + self._transition[prev, state, obs_ix - 1]
                    if value > best_val:
                        best_val = value
                        best_state = prev
                col_b[state] = best_val + emission
                prev_matrix[state, obs_ix] = best_state

        state = np.argmax(col_b)
        final_prob = col_b[state]
        states_vec = np.zeros(self._n_observations, dtype=np.int16)
        for obs_ix in reversed(range(self._n_observations)):
            states_vec[obs_ix] = state
            state = prev_matrix[state, obs_ix]
        return final_prob, states_vec

    def viterbi_many(self):
        """
        Returns
            - List of log probabilities (n_samples),
            - Matrix of best hidden states (n_samples x n_observations).
        """
        probs = np.full(self._n_samples, np.nan)
        states_matrix = np.zeros((self._n_samples, self._n_observations), dtype=np.int16)

        for sample_id in range(self._n_samples):
            sample_prob, states_vec = self.viterbi(sample_id)
            probs[sample_id] = sample_prob
            states_matrix[sample_id, :] = states_vec
        return probs, states_matrix

    def path_likelihood(self, sample_id, path):
        """
        Returns log-probability of the path, where path is a list of segments with attributes
        start_ix, end_ix and state.
        Should call run_forward_backward beforehand.
        """
        assert self._total_probs is not None
        alpha = self._alphas[sample_id]
        beta = self._betas[sample_id]
        emission_matrix = self._emission_matrices[sample_id]

        prob = -self._total_probs[sample_id]
        first_segment = path[0]
        prob += alpha[first_segment.state, first_segment.start_ix]
        last_segment = path[-1]
        prob += beta[last_segment.state, last_segment.end_ix - 1]
        if not np.isfinite(prob):
            return prob

        prev = None
        for segment in path:
            if prev is not None:
                assert prev.end_ix == segment.start_ix
                prob += self._transition[prev.state, segment.state, segment.start_ix - 1] \
                    + emission_matrix[segment.state, segment.start_ix]
            prob += np.sum(self._transition[segment.state, segment.state, segment.start_ix : segment.end_ix - 1])
            prob += np.sum(emission_matrix[segment.state, segment.start_ix + 1 : segment.end_ix])
            prev = segment
        return prob

    def _forward_backward(self, sample_id, possible_states=None):
        """
        Returns two matrices in natural log space: alpha and beta (both n_states x n_observations).
        """
        if possible_states is not None:
            a0, b0 = possible_states[sample_id]
        else:
            a0 = 0
            b0 = self._n_hidden

        emission_matrix = self._emission_matrices[sample_id]
        shape = (self._n_hidden, self._n_observations)
        alpha = np.full(shape, -np.inf, dtype=np.float64)
        alpha[:, 0] = self._initial + emission_matrix[:, 0]
        for obs_ix in range(1, self._n_observations):
            for state in range(a0, b0):
                emission = emission_matrix[state, obs_ix]
                if not np.isfinite(emission):
                    continue
                a, b = self.consecutive_state_range(state)
                a = max(a, a0)
                b = min(b, b0)
                alpha[state, obs_ix] = emission \
                    + logsumexp(alpha[a : b, obs_ix - 1] + self._transition[a : b, state, obs_ix - 1])

        beta = np.full(shape, -np.inf, dtype=np.float64)
        beta[:, self._n_observations - 1] = 0.0
        for obs_ix in reversed(range(self._n_observations - 1)):
            for state in range(a0, b0):
                a, b = self.consecutive_state_range(state)
                a = max(a, a0)
                b = min(b, b0)
                beta[state, obs_ix] = logsumexp(emission_matrix[a : b, obs_ix + 1] + beta[a : b, obs_ix + 1] +
                    self._transition[state, a : b, obs_ix])
        return alpha, beta

    def run_forward_backward(self, possible_states=None):
        ret_shape = (self._n_samples, self._n_hidden, self._n_observations)
        alphas = np.full(ret_shape, np.nan)
        betas = np.full(ret_shape, np.nan)
        gammas = np.full(ret_shape, np.nan)

        for sample_id in range(self._n_samples):
            alpha, beta = self._forward_backward(sample_id, possible_states=possible_states)
            sum_prob = logsumexp(alpha[:, self._n_observations - 1])
            alphas[sample_id] = alpha
            betas[sample_id] = beta
            gammas[sample_id] = alpha + beta - sum_prob

        self._total_probs = logsumexp(alphas[:, :, self._n_observations - 1], axis=1)
        self._alphas = alphas
        self._betas = betas
        self._gammas = gammas


class CopyNumHmm(HmmModel):
    def __init__(self, n_samples, ref_copy_num, left_states, right_states, n_observations, max_state_dist):
        super().__init__(n_samples, left_states + right_states + 1, n_observations, max_state_dist=max_state_dist)
        self._ref_copy_num = ref_copy_num
        self._left_states = left_states
        self._right_states = right_states

    @property
    def ref_copy_num(self):
        return self._ref_copy_num

    @property
    def left_states(self):
        return self._left_states

    @property
    def right_states(self):
        return self._right_states

    def set_transition_from_jumps(self, jump_probs):
        assert jump_probs.shape[0] == 2 * self._max_state_dist + 1
        if jump_probs.ndim == 1:
            transition = np.full((self._n_hidden, self._n_hidden), -np.inf)
        else:
            assert jump_probs.shape[1] == self._n_observations - 1
            transition = np.full((self._n_hidden, self._n_hidden, self._n_observations - 1), -np.inf)

        for state1 in range(self._n_hidden):
            for state2 in range(*self.consecutive_state_range(state1)):
                transition[state1, state2] = jump_probs[state2 - state1 + self._max_state_dist]
            transition[state1] -= logsumexp(transition[state1], axis=0)
        self.set_transition(transition)

    def set_const_transitions(self, jump1):
        jump_probs = _transform_jump_probs(down=jump1, up=jump1, min_prob=-np.inf, max_prob=0,
            max_state_dist=self._max_state_dist)
        self.set_transition_from_jumps(jump_probs)

    def get_copy_num(self, state):
        return state - self.left_states + self._ref_copy_num

    def get_state(self, pred_cn):
        return pred_cn - self._ref_copy_num + self.left_states

    def format_cn(self, pred_cn):
        diff = pred_cn - self._ref_copy_num
        if -diff == self._left_states and pred_cn != 0:
            return '<{}'.format(pred_cn + 1)
        if diff == self._right_states:
            return '>{}'.format(pred_cn - 1)
        return str(pred_cn)

    def middle_state_transitions(self):
        """
        Returns matrix (n_observations - 1  x  3), which stores transition probabilities (one down, same, one up).
        """
        state = self._n_hidden // 2
        res = np.full((self._n_observations - 1, 3), -np.inf)
        for snext in range(state - 1, state + 2):
            res[:, snext - state + 1] = self._transition[state, snext, :]
        return res

    def calculate_emission_matrices(self, depth_matrix, windows, multipliers, mult_weights, bg_depth, error_rate=0.01):
        assert depth_matrix.shape == (self._n_observations, self._n_samples)
        state_ploidies = self.get_copy_num(np.arange(self._n_hidden)).astype(np.float)
        assert state_ploidies[0] >= 0.0
        state_ploidies[0] = max(state_ploidies[0], error_rate)

        emission_matrices = np.full((self._n_samples, self._n_hidden, self._n_observations), -np.inf)
        for sample_id in range(self._n_samples):
            observations = depth_matrix[:, sample_id]

            for obs_ix, window in enumerate(windows):
                mult = multipliers[obs_ix]
                mult_weight = mult_weights[obs_ix]
                if mult_weight == 0.0:
                    emission_matrices[sample_id, :, obs_ix] = -np.log(self._n_hidden)
                    continue

                n_param, p_param = bg_depth.at(sample_id, window.gc_content)
                n_params = 0.5 * n_param * state_ploidies * mult
                curr_emissions = mult_weight * nbinom.logpmf(observations[obs_ix], n_params, p_param)
                emission_matrices[sample_id, :, obs_ix] = curr_emissions - logsumexp(curr_emissions)
        self.set_emission_matrices(emission_matrices)

    def baum_welch_initial(self, min_prob):
        log_n_samples = np.log(self._n_samples)
        initial = logsumexp(self._gammas[:, :, 0], axis=0) - log_n_samples
        initial = np.maximum(initial, min_prob)
        initial -= logsumexp(initial)
        self.set_initial(initial)

    def baum_welch_transitions(self):
        """
        Returns matrix of size (n_observations - 1, 2): probabilities of going down, and of going up.
        """
        assert self._total_probs is not None
        log_n_samples = np.log(self._n_samples)

        state_dist = self._max_state_dist
        down_up = np.full((self._n_observations - 1, 2), np.nan)
        for obs_ix in range(self._n_observations - 1):
            jump_probs = np.zeros(2 * state_dist)
            # We do not need to calculate probabilities for one of the jumps (here, we skip 0),
            # because it can be inferred from others.
            for i in range(2 * state_dist):
                # i = [0, 1, 2, 3] transforms into jumps = [-2, -1, 1, 2].
                jump = i - state_dist + (i // state_dist)
                # If jump is negative, we want to have higher starting point.
                curr_a = max(0, -jump)
                # If jump is positive, we want to have lower ending point.
                curr_b = min(self._n_hidden, self._n_hidden - jump)
                next_a = curr_a + jump
                next_b = curr_b + jump
                assert curr_b - curr_a > 0 and next_a >= 0 and next_b <= self._n_hidden
                curr_range = np.arange(curr_a, curr_b)
                next_range = np.arange(next_a, next_b)

                # shape == (curr_b - curr_a,)
                transitions_vec = self._transition[curr_range, next_range, obs_ix]
                # All next shapes == (n_samples, curr_b - curr_a)
                next_emissions = self._emission_matrices[:, next_range, obs_ix + 1]
                curr_alphas = self._alphas[:, curr_range, obs_ix]
                next_betas = self._betas[:, next_range, obs_ix + 1]

                summands = curr_alphas + next_betas + transitions_vec[np.newaxis, :] + next_emissions
                prob = logsumexp(logsumexp(summands, axis=1) - self._total_probs) - log_n_samples
                jump_probs[i] = prob
            down_up[obs_ix, 0] = logsumexp(jump_probs[:state_dist])
            down_up[obs_ix, 1] = logsumexp(jump_probs[state_dist:])
        return down_up

    def recalculate_transition_probs(self, down_up, min_prob, max_prob):
        """
        Transforms probabilities of going down, staying the same, or going up, into transition probabilities.
        min_rob and max_prob should be logarithmic and only apply to single jumps up and down.
        """
        new_down_up = np.zeros_like(down_up)
        new_down_up[:, 0] = _find_peaks(down_up[:, 0], min_prob)
        new_down_up[:, 1] = _find_peaks(down_up[:, 1], min_prob)

        jump_probs = np.zeros((2 * self._max_state_dist + 1, self._n_observations - 1))
        for obs_ix in range(self._n_observations - 1):
            down, up = new_down_up[obs_ix]
            jump_probs[:, obs_ix] = _transform_jump_probs(down, up, min_prob, max_prob, self._max_state_dist)
        self.set_transition_from_jumps(jump_probs)


def _find_peaks(vec, min_prob, distance=10):
    peaks, _ = signal.find_peaks(vec, height=min_prob, distance=distance)
    n_peaks = len(peaks)
    n = len(vec)

    res = np.copy(vec)
    prev_pos = 0
    for i, pos in enumerate(peaks):
        left = max(prev_pos, pos - distance)
        right = min(n, pos + distance)
        if i + 1 < n_peaks:
            next_pos = peaks[i + 1]
            right = min(right, (pos + next_pos) // 2)
        res[left : right] = -np.inf
        res[pos] = logsumexp(vec[left : right])
        prev_pos = right
    return res


def _transform_jump_probs(down, up, min_prob, max_prob, max_state_dist):
    res = np.full(max_state_dist * 2 + 1, -np.inf)
    down = np.clip(down, min_prob, max_prob)
    up = np.clip(up, min_prob, max_prob)
    for i in range(1, max_state_dist + 1):
        res[max_state_dist - i] = down * i
        res[max_state_dist + i] = up * i
    res[max_state_dist] = logsumexp((0, logsumexp(res)), b=(1, -1))
    return res


def _narrow_possible_states(prob_matrices, min_prev_prob=np.log(0.01), margin=1):
    """
    Returns matrix (n_samples x 2), that stores range of possible states for each sample.
    """
    n_samples, n_hidden, _n_observations = prob_matrices.shape
    assert n_hidden < 256
    assert min_prev_prob < 0

    res = np.zeros((n_samples, 2), dtype=np.uint16)
    for sample_id, prob_matrix in enumerate(prob_matrices):
        res[:, 1] = n_hidden
        possible_states = np.where(np.max(prob_matrix, axis=1) >= min_prev_prob)[0]
        if len(possible_states):
            res[sample_id, 0] = max(0, possible_states[0] - margin)
            res[sample_id, 1] = min(n_hidden, possible_states[-1] + margin)
    return res


SimpleSegment = collections.namedtuple('SimpleSegment', 'start_ix end_ix state')

def get_simple_path(states_vec, subregions=None):
    """
    Returns tuple of tuples (start_ix, end_ix, state).
    """
    if subregions is None:
        subregions = ((0, len(states_vec)),)
    res = []
    for subr_start, subr_end in subregions:
        start = subr_start
        for i in range(subr_start + 1, subr_end):
            if states_vec[i - 1] != states_vec[i]:
                res.append(SimpleSegment(start, i, states_vec[start]))
                start = i
        if start < subr_end:
            res.append(SimpleSegment(start, subr_end, states_vec[start]))
    return tuple(res)


def _minus_value_len(key_value):
    return -len(key_value[1])


class PathSegment:
    def __init__(self, start_ix, end_ix, state, model, windows, const_regions_extra):
        self.start_ix = start_ix
        self.end_ix = end_ix
        self.state = state
        self.cn = model.get_copy_num(state)

        start_window = windows[start_ix]
        end_window = windows[end_ix - 1]
        const_region, reg_start_ix, reg_end_ix = const_regions_extra[start_window.const_region_ix]
        assert const_region.ix == end_window.const_region_ix
        if start_ix == reg_start_ix:
            start_window = const_region
        if end_ix == reg_end_ix:
            end_window = const_region

        combined = start_window.continue_if_possible(end_window, max_dist=sys.maxsize, strict_order=False)
        assert combined is not None
        self.dupl_region = cn_tools.DuplRegion(None, *combined)
        self.const_region_ix = const_region.ix
        self.group_name = const_region.group_name

    def crop(self, template_segment):
        if self.end_ix <= template_segment.start_ix or self.start_ix >= template_segment.end_ix:
            return None
        cls = self.__class__
        res = cls.__new__(cls)
        res.start_ix = max(self.start_ix, template_segment.start_ix)
        res.end_ix = min(self.end_ix, template_segment.end_ix)
        res.state = self.state
        res.cn = self.cn
        res.const_region_ix = self.const_region_ix
        res.group_name = self.group_name

        start_window = self.dupl_region if self.start_ix > template_segment.start_ix else template_segment.dupl_region
        end_window = self.dupl_region if self.end_ix < template_segment.end_ix else template_segment.dupl_region
        combined = start_window.continue_if_possible(end_window, max_dist=sys.maxsize, strict_order=False)
        assert combined is not None
        res.dupl_region = cn_tools.DuplRegion(None, *combined)
        return res

    def __len__(self):
        return self.end_ix - self.start_ix

    def __str__(self):
        return 'PathSegment(start={}, end={}, state={}, copy_num={})'.format(self.start_ix, self.end_ix,
            self.state, self.cn)


class Path:
    def __init__(self, path_segments, sample_ids):
        self.segments = path_segments
        self.sample_ids = sample_ids
        self.weight = None
        self._assign_name()

    def _assign_name(self):
        cn_occur = collections.Counter()
        for segment in self.segments:
            cn_occur[segment.cn] += len(segment.dupl_region.region1)
        cn = cn_occur.most_common(1)[0][0]
        n_ins = 0
        n_del = 0
        for i, segment in enumerate(self.segments):
            if segment.cn != cn and (i == 0 or self.segments[i - 1].cn != segment.cn):
                if segment.cn > cn:
                    n_ins += 1
                else:
                    n_del += 1
        self.name = 'cn{}'.format(cn)
        if n_ins == 0 and n_del == 0:
            return

        self.name += '_'
        if n_ins:
            self.name += '{}i'.format(n_ins)
        if n_del:
            self.name += '{}d'.format(n_del)

    def calculate_weight(self, n_samples, min_weight=0.001):
        if self.weight is not None:
            return

        if len({ segment.state for segment in self.segments }) == 1:
            self.weight = 1.0
        else:
            self.weight = max(len(self.sample_ids) / n_samples, min_weight)

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, index):
        return self.segments.__getitem__(index)

    def __len__(self):
        return len(self.segments)


def _extract_paths(states_matrix, model, windows, const_regions, model_params):
    """
    Returns two lists:
        - paths: dictionary {name: Path},
        - sample_paths: list of path names, one for each sample.
    """
    n_samples, n_observations = states_matrix.shape
    const_regions_extra = {}
    obs_subregions = []
    for i, window in enumerate(windows):
        region_ix = window.const_region_ix
        if region_ix not in const_regions_extra:
            const_regions_extra[region_ix] = [const_regions[region_ix], i, i + 1]
            obs_subregions.append([i, i + 1])
        else:
            const_regions_extra[region_ix][2] = i + 1
            obs_subregions[-1][1] = i + 1

    pre_paths = collections.defaultdict(list)
    in_path_info = {}
    if model_params.is_loaded:
        group_name = next(iter(const_regions_extra.values()))[0].group_name
        for path_name, path_str in model_params.get_hmm_paths(group_name):
            path_str = path_str.split(',')
            weight = float(path_str[0])
            pre_path = []
            for entry in path_str[1:]:
                start_ix, end_ix, entry_cn = map(int, entry.split('-'))
                pre_path.append(SimpleSegment(start_ix, end_ix, model.get_state(entry_cn)))
            pre_path = tuple(pre_path)
            pre_paths[pre_path] = []
            in_path_info[pre_path] = (path_name, weight)
    else:
        min_const_state = np.clip(np.min(states_matrix) - 1, a_min=0, a_max=model.left_states)
        max_const_state = np.clip(np.max(states_matrix) + 1, a_min=model.left_states, a_max=model.n_hidden - 1)
        for state in range(min_const_state, max_const_state + 1):
            const_path = []
            for start, end in obs_subregions:
                const_path.append((start, end, state))
            pre_paths[tuple(const_path)] = []

    for sample_id, row in enumerate(states_matrix):
        pre_path = get_simple_path(row, obs_subregions)
        pre_paths[pre_path].append(sample_id)

    names_counter = collections.Counter()
    sample_paths = [None] * n_samples
    paths = {}
    for pre_path, sample_ids in sorted(pre_paths.items(), key=_minus_value_len):
        segments = []
        for start_ix, end_ix, state in pre_path:
            segments.append(PathSegment(start_ix, end_ix, state, model, windows, const_regions_extra))
        path = Path(segments, sample_ids)

        if model_params.is_loaded:
            path_info = in_path_info.get(pre_path)
            if path_info is None:
                path.name += '_new'
                path.weight = 0.0
            else:
                path.name, path.weight = path_info
        else:
            path.calculate_weight(n_samples)

        init_name = path.name
        if init_name in names_counter:
            path.name += '_' + common.letter_suffix(names_counter[init_name])
        names_counter[init_name] += 1
        paths[path.name] = path
        for sample_id in sample_ids:
            sample_paths[sample_id] = path.name
    return const_regions_extra, paths, sample_paths


def _summarize_paths(dupl_hierarchy, windows, paths, genome, samples, summary_out):
    n_observations = len(windows)
    group_name = dupl_hierarchy.window_group_name(windows[0])
    region_group = dupl_hierarchy.get_group(group_name)
    cn = region_group.cn

    summary_out.write('Region group {}:\n'.format(group_name))
    for region_ix in region_group.region_ixs:
        region = dupl_hierarchy.const_regions[region_ix].region1
        summary_out.write('    {} ({:7,} bp)\n'.format(region.to_str_comma(genome), len(region)))
    summary_out.write('    Copy number: {}\n'.format(cn))
    summary_out.write('    HMM windows: {}\n'.format(n_observations))

    for path in paths.values():
        summary_out.write('    ===============\n')
        summary_out.write('    Path  "{}"  (weight {:.3f})\n'.format(path.name, path.weight))
        if path.sample_ids:
            summary_out.write('        {} samples: {}\n'.format(len(path.sample_ids),
                ' '.join(samples[i] for i in path.sample_ids)))
        else:
            summary_out.write('        no samples\n')

        summary_out.write('    {} segments:\n'.format(len(path)))
        for segment in path:
            segm_region = segment.dupl_region.region1
            summary_out.write('        {} ({:7,} bp),  copy num {},  windows {}-{}\n'.format(
                segm_region.to_str_comma(genome), len(segm_region),
                segment.cn, windows[segment.start_ix].ix, windows[segment.end_ix - 1].ix))


def _optimize_multipliers(nbinom_params, obs_depth, copy_num_probs, copy_num_range, mult_bounds):
    n_samples = len(obs_depth)
    range_size = len(copy_num_range)
    assert nbinom_params.shape == (n_samples, 2)
    assert copy_num_probs.shape == (range_size, n_samples)
    assert copy_num_range[0] > 0
    nbinom_n_matrix = nbinom_params[:, 0] * copy_num_range[:, np.newaxis] / 2
    nbinom_ps = nbinom_params[:, 1]

    def inner(mult):
        matr = nbinom.logpmf(obs_depth, nbinom_n_matrix * mult, nbinom_ps) + copy_num_probs
        res = -np.sum(logsumexp(matr, axis=0))
        return res

    x0 = 1.0
    lik0 = inner(x0)
    sol = optimize.minimize_scalar(inner, bounds=mult_bounds, method='bounded')
    return x0, lik0, sol.x, sol.fun


def _find_multipliers(depth_matrix, windows, bg_depth, model, min_samples, edge_prob=0.1, sample_ids=None):
    if sample_ids is None:
        sample_ids = np.arange(model.n_samples)
        sample_ids_slice = slice(0, model.n_samples)
    else:
        sample_ids_slice = sample_ids

    # Multipliers with extreme values will be discarded later.
    mult_bounds = (0.4, 2.1)
    edge_log_prob = np.log(edge_prob)

    n_hidden = model.n_hidden
    copy_num_range = model.get_copy_num(np.arange(1, n_hidden - 1))
    prob_matrices = model.gammas

    multipliers = np.ones(len(windows))
    for obs_ix, window in enumerate(windows):
        # Keep samples, for which there is only a little probability of having smallest or largest available copy number.
        # We do this because in reality these samples may have even lower or higher copy number.
        edge_probs = np.max(prob_matrices[sample_ids_slice][:, (0, n_hidden - 1), obs_ix], axis=1)
        curr_sample_ids = sample_ids[edge_probs < edge_log_prob]
        if len(curr_sample_ids) < min_samples:
            continue

        obs_depth = depth_matrix[obs_ix, curr_sample_ids]
        nbinom_params = bg_depth.at(curr_sample_ids, window.gc_content)
        copy_num_probs = prob_matrices[curr_sample_ids, 1:n_hidden - 1, obs_ix].T
        mult0, lik0, mult1, lik1 = _optimize_multipliers(nbinom_params, obs_depth, copy_num_probs,
            copy_num_range, mult_bounds)
        multipliers[obs_ix] = mult1
    return multipliers


def _setup_model(depth_matrix, windows, bg_depth, copy_num_range, copy_num_jump, min_trans_prob):
    n_observations, n_samples = depth_matrix.shape
    ref_copy_num = windows[0].cn
    left_states, right_states = _select_hidden_states(depth_matrix, windows, bg_depth, ref_copy_num, copy_num_range)
    model = CopyNumHmm(n_samples, ref_copy_num, left_states, right_states, n_observations, copy_num_jump)

    # Set prior for the reference copy number on the first iteration.
    initial = np.full(model.n_hidden, min_trans_prob / 2)
    initial[model.left_states] = 0.0
    initial -= logsumexp(initial)
    model.set_initial(initial)

    model.set_const_transitions(min_trans_prob)
    multipliers = np.ones(model.n_observations)
    mult_weights = _get_mult_weights(multipliers, ref_copy_num)
    model.calculate_emission_matrices(depth_matrix, windows, multipliers, mult_weights, bg_depth)
    return model, multipliers


def _load_joint_model(depth_matrix, windows, window_ixs, bg_depth, group_name, model_params,
        copy_num_jump, use_multipliers):
    n_observations, n_samples = depth_matrix.shape
    ref_copy_num = windows[0].cn

    joint_entry = model_params.get_hmm_entry(group_name)
    min_cn, max_cn = map(int, joint_entry.info['copy_nums'].split(','))
    left_states = ref_copy_num - min_cn
    right_states = max_cn - ref_copy_num
    model = CopyNumHmm(n_samples, ref_copy_num, left_states, right_states, n_observations, copy_num_jump)

    initial = -np.array(list(map(float, joint_entry.info['initial'].split(','))))
    initial -= logsumexp(initial)
    model.set_initial(initial)

    multipliers, jump_probs = model_params.get_hmm_data(window_ixs, model.max_state_dist)
    down = jump_probs[:, 0]
    up = jump_probs[:, 2]
    jump_probs = np.zeros((2 * model.max_state_dist + 1, n_observations - 1))
    for obs_ix in range(n_observations - 1):
        jump_probs[:, obs_ix] = _transform_jump_probs(down[obs_ix], up[obs_ix], -np.inf, 0, model.max_state_dist)
    model.set_transition_from_jumps(jump_probs)

    if not use_multipliers:
        multipliers = np.ones(n_observations)
    mult_weights = _get_mult_weights(multipliers, model.ref_copy_num)
    model.calculate_emission_matrices(depth_matrix, windows, multipliers, mult_weights, bg_depth)
    return model, multipliers


def _write_hmm_params(group_name, iteration, windows, multipliers, mult_weights, down_up, model, params_out):
    copy_numbers = model.get_copy_num(np.arange(model.n_hidden))
    initial = model.initial
    params_out.write('# Initial for group {} iteration {}: {}\n'.format(group_name, iteration,
        '\t'.join(map('%d=%.4f'.__mod__, zip(copy_numbers, initial)))))

    jump_probs_ixs = slice(model.max_state_dist - 1, model.max_state_dist + 2)
    jump_probs = model.middle_state_transitions()
    n_observations = len(windows)
    for obs_ix, window in enumerate(windows):
        params_out.write('{}\t{}\t{}\t{:.4f}\t{:.4f}\t'.format(group_name, iteration, windows[obs_ix].ix,
            multipliers[obs_ix], mult_weights[obs_ix]))
        if obs_ix == n_observations - 1:
            params_out.write('\t'.join(('NA',) * 7))
        else:
            params_out.write('{:.3f}\t{:.3f}\t'.format(*(down_up[obs_ix] / common.LOG10)))
            params_out.write('\t'.join(map('{:.3f}'.format, jump_probs[obs_ix] / common.LOG10)))
        params_out.write('\n')


def _prob_matrices_to_states(model):
    prob_matrices = model.gammas
    n_samples, n_hidden, n_observations = prob_matrices.shape
    copy_numbers = model.get_copy_num(np.arange(n_hidden))
    states_matrix = np.zeros((n_samples, n_observations))

    for obs_ix in range(n_observations):
        for sample_id in range(n_samples):
            sample_vec = prob_matrices[sample_id, :, obs_ix]
            states_matrix[sample_id, obs_ix] = np.sum(np.exp(sample_vec) * copy_numbers)
    return states_matrix


def _write_states_matrix(states_matrix, window_ixs, out_prefix, out, fmt='{}'):
    for obs_ix, window_ix in enumerate(window_ixs):
        out.write('{}\t{}\t'.format(out_prefix, window_ix))
        out.write('\t'.join(map(fmt.format, states_matrix[:, obs_ix])))
        out.write('\n')


def _get_mult_weights(multipliers, ref_copy_num):
    # For reference copy number = 4 this will transform 5/4 -> 1 and 3/4 -> -1.
    mult_weights = (multipliers - 1.0) * ref_copy_num
    return common.tricube_kernel(mult_weights)


def _single_iteration(depth_matrix, windows, bg_depth, model, group_name, iteration, params_out,
        possible_states=None, *, min_samples, min_trans_prob, max_trans_prob, use_multipliers):
    model.run_forward_backward(possible_states)
    total_prob = np.sum(model.total_probs)
    common.log('        Iteration {:>2}:   Likelihood {:,.3f}'.format(iteration, total_prob / common.LOG10))

    down_up = model.baum_welch_transitions()
    n_samples = depth_matrix.shape[1]
    min_initial_prob = max(min_trans_prob / 2, -np.log(n_samples))
    model.baum_welch_initial(min_initial_prob)
    model.recalculate_transition_probs(down_up, min_trans_prob, max_trans_prob)
    if use_multipliers:
        multipliers = _find_multipliers(depth_matrix, windows, bg_depth, model, min_samples)
        mult_weights = _get_mult_weights(multipliers, model.ref_copy_num)
    else:
        multipliers = np.ones(model.n_observations)
        mult_weights = np.ones(model.n_observations)

    _write_hmm_params(group_name, iteration, windows, multipliers, mult_weights, down_up, model, params_out)
    model.calculate_emission_matrices(depth_matrix, windows, multipliers, mult_weights, bg_depth)
    params_out.write('# Likelihood for group {} iteration {}: {:.5f}\n'.format(
        group_name, iteration, total_prob / common.LOG10))
    return total_prob, multipliers


def _select_hidden_states(depth_matrix, windows, bg_depth, ref_copy_num, copy_num_range):
    n_observations, n_samples = depth_matrix.shape
    norm_depth = np.zeros((n_observations, n_samples))

    for obs_ix, window in enumerate(windows):
        nbinom_params = bg_depth.at(slice(None), window.gc_content)
        nbinom_n = nbinom_params[:, 0]
        nbinom_p = nbinom_params[:, 1]
        mean_depth = 0.5 * (1 - nbinom_p) * nbinom_n / nbinom_p
        norm_depth[obs_ix] = depth_matrix[obs_ix] / mean_depth

    norm_means = np.mean(norm_depth, axis=0)
    min_value = np.min(norm_means)
    min_value = max(int(np.floor(min_value) - 1), ref_copy_num - 2 * copy_num_range[0])
    min_copy_num = max(0, min(ref_copy_num - copy_num_range[0], min_value))

    max_value = np.max(norm_means)
    max_value = min(int(np.ceil(max_value) + 1), ref_copy_num + 2 * copy_num_range[1])
    max_copy_num = max(ref_copy_num + copy_num_range[1], max_value)

    left_states = ref_copy_num - max(0, min_copy_num - 1)
    right_states = max_copy_num - ref_copy_num + 1
    return left_states, right_states


def find_cn_profiles(region_group_extra, full_depth_matrix, samples, bg_depth, genome, out, model_params, *,
        min_samples, copy_num_range, copy_num_jump, min_trans_prob, max_trans_prob=np.log(0.1), use_multipliers=True):
    dupl_hierarchy = region_group_extra.dupl_hierarchy
    windows = region_group_extra.viterbi_windows
    ref_copy_num = windows[0].cn
    window_ixs = np.array([window.ix for window in windows])
    n_observations = len(windows)
    n_samples = len(samples)
    region_group = region_group_extra.region_group
    group_name = region_group.name

    depth_matrix = full_depth_matrix[window_ixs, :]
    if model_params.is_loaded:
        model, multipliers = _load_joint_model(depth_matrix, windows, window_ixs, bg_depth, group_name, model_params,
            copy_num_jump, use_multipliers)
        single_iter = True
    else:
        model, multipliers = _setup_model(depth_matrix, windows, bg_depth, copy_num_range, copy_num_jump, min_trans_prob)
        assert np.all(multipliers == 1)
        single_iter = n_samples < min_samples
    common.log('    Calculating copy number profiles within range [{}, {}]'.format(
        model.get_copy_num(0), model.get_copy_num(model.n_hidden - 1)))

    prev_prob = -np.inf
    prev_pseudo_states_matrix = None
    possible_states = None

    if single_iter:
        mult_weights = _get_mult_weights(multipliers, ref_copy_num)
        down_up = np.full((n_observations - 1, 2), np.nan)
        _write_hmm_params(group_name, 0, windows, multipliers, mult_weights, down_up, model, out.hmm_params)

    ITERATIONS = 50
    # Do not run this for if there are not enough samples, or if the model was loaded from a previous run.
    for iteration in range(1, 0 if single_iter else ITERATIONS + 1):
        prob, multipliers = _single_iteration(depth_matrix, windows, bg_depth, model,
            group_name, iteration, out.hmm_params, possible_states=possible_states, min_samples=min_samples,
            min_trans_prob=min_trans_prob, max_trans_prob=max_trans_prob, use_multipliers=use_multipliers)
        possible_states = _narrow_possible_states(model.gammas)

        pseudo_states_matrix = _prob_matrices_to_states(model)
        out_prefix = '{}\t{}'.format(group_name, iteration)
        _write_states_matrix(pseudo_states_matrix, window_ixs, out_prefix, out.hmm_states, fmt='{:.4g}')

        same_result = prev_pseudo_states_matrix is not None \
            and np.all(pseudo_states_matrix.round() == prev_pseudo_states_matrix.round())

        rel_improv = (prob - prev_prob) / np.abs(prev_prob) if np.isfinite(prev_prob) else 1
        improv = prob - prev_prob if np.isfinite(prev_prob) else 1
        if not np.isfinite(prob):
            common.log('Warning: HMM likelihood is infinite')
            rel_improv = -1
            improv = -1
        # Require different improvement based on the number of iterations.
        if (iteration > 2 and rel_improv <= 0 and same_result) \
                or (iteration > 2 and improv <= 0.01 and same_result) \
                or (iteration > 10 and rel_improv <= 1e-8) \
                or (iteration > 20 and rel_improv <= 1e-6):
            break
        prev_prob = prob
        prev_pseudo_states_matrix = pseudo_states_matrix

    viterbi_prob, states_matrix = model.viterbi_many()
    _write_states_matrix(model.get_copy_num(states_matrix), window_ixs, group_name + '\tv', out.hmm_states)
    const_regions_extra, paths, sample_paths = _extract_paths(states_matrix, model, windows,
        dupl_hierarchy.const_regions, model_params)
    _summarize_paths(dupl_hierarchy, windows, paths, genome, samples, out.viterbi_summary)

    model.run_forward_backward(possible_states=possible_states)
    common.log('    Writing detailed copy number profiles')
    _write_detailed_cn(model, samples, windows, genome, out.detailed_cn)

    common.log('    Finalizing total copy number predictions')
    sample_const_regions = []
    sample_reliable_regions = []
    for sample_id in range(n_samples):
        sample_path = paths[sample_paths[sample_id]]
        sample_path = _split_path_by_probs(sample_path, model.gammas[sample_id], model, windows, const_regions_extra)

        sample_const_regions.append(_get_sample_const_regions(sample_id, sample_path, model))
        sample_reliable_regions.append(_get_sample_reliable_region(sample_path))
    region_group_extra.set_viterbi_res(sample_const_regions, sample_reliable_regions)

    if not model_params.is_loaded:
        common.log('    Saving HMM parameters to joint data')
        model_params.set_hmm_results(region_group, window_ixs, model, multipliers, paths)


def _write_detailed_cn(model, samples, windows, genome, out):
    prob_matrices = model.gammas
    for i, window in enumerate(windows):
        out.write(window.region1.to_bed(genome))
        out.write('\t{}\t'.format(window.ix))
        out.write(window.regions2_str(genome))
        for sample_id, sample in enumerate(samples):
            probs = prob_matrices[sample_id, :, i]
            probs_ixs = np.argsort(probs)
            best_state = probs_ixs[-1]
            second_state = probs_ixs[-2]

            best_cn = model.get_copy_num(best_state)
            second_cn = model.get_copy_num(second_state)
            out.write('\t{}:{:.3f} {}:{:.3f}'.format(
                model.format_cn(best_cn), abs(probs[best_state] / common.LOG10),
                model.format_cn(second_cn), abs(probs[second_state] / common.LOG10)))
        out.write('\n')


def write_headers(out, samples):
    samples_str = '\t'.join(samples)
    out.checked_write('hmm_states', 'region_group\titeration\twindow_ix\t{}\n'.format(samples_str))
    out.checked_write('hmm_params', 'region_group\titeration\twindow_ix\tmultiplier\tmult_weight\t'
        'down_bw\tup_bw\ttrans_down1\ttrans_same\ttrans_up1\n')
    out.checked_write('detailed_cn',
        '## Cell format: best_copy_num:-log10(prob) second_best:-log10(prob).\n',
        '#chrom\tstart\tend\twindow_ix\thomologous_regions\t{}\n'.format(samples_str))


def _split_path_by_probs(path, prob_matrix, model, windows, const_regions_extra, min_len=10, thresh_prob=np.log(0.99)):
    """
    Splits path segments into subsegments if there is a drop in probability longer than min_len.
    """
    new_path = []
    for segment in path:
        if len(segment) < min_len:
            new_path.append(segment)
            continue

        over_thresh = prob_matrix[segment.state, segment.start_ix : segment.end_ix] >= thresh_prob
        over_thresh_path = get_simple_path(over_thresh)

        changed = True
        # Simplify the path by removing any regions shorter than min_len.
        while changed and len(over_thresh_path) > 1:
            changed = False
            for subsegment in over_thresh_path:
                if subsegment.end_ix - subsegment.start_ix < min_len:
                    over_thresh[subsegment.start_ix : subsegment.end_ix] = not subsegment.state
                    changed = True
                    break
            if changed:
                over_thresh_path = get_simple_path(over_thresh)

        if len(over_thresh_path) == 1:
            new_path.append(segment)
            continue

        for subsegment in over_thresh_path:
            new_segment = PathSegment(segment.start_ix + subsegment.start_ix, segment.start_ix + subsegment.end_ix,
                segment.state, model, windows, const_regions_extra)
            new_path.append(new_segment)
    assert sum(map(len, path)) == sum(map(len, new_path))
    return new_path


def _get_sample_const_regions(sample_id, main_path, model):
    res = []
    for segment in main_path:
        probs = [model.path_likelihood(sample_id, (segment,))]
        cns = [model.get_copy_num(segment.state)]
        for state in range(max(0, segment.state - 2), min(model.n_hidden, segment.state + 3)):
            if state != segment.state:
                curr_path = (SimpleSegment(segment.start_ix, segment.end_ix, state),)
                probs.append(model.path_likelihood(sample_id, curr_path))
                cns.append(model.get_copy_num(state))

        probs = np.array(probs)
        probs -= logsumexp(probs)
        qual = common.phred_qual(probs, best_ix=0)
        dupl_region = segment.dupl_region

        pred_cn = cns[0]
        pred_cn_str = model.format_cn(pred_cn)
        cn_pred = cn_tools.CopyNumPrediction(dupl_region.region1, dupl_region.regions2, pred_cn, pred_cn_str, qual)
        cn_pred.info['region_ix'] = segment.const_region_ix

        if qual <= 40 and len(probs) > 1:
            probs = np.abs(probs)
            ixs = np.argsort(probs)
            cn_pred.info['agCN_probs'] = ','.join(
                '{}:{:.3g}'.format(model.format_cn(cns[i]), probs[i] / common.LOG10) for i in ixs if probs[i] < 11)
        res.append(cn_pred)
    return res


def _get_sample_reliable_region(path, min_reliable_windows=5):
    best_region = None
    best_key = (0, 0)
    rel_regions = [[]]
    for segment in path:
        if segment.cn == segment.dupl_region.cn:
            rel_regions[-1].append(segment)
        else:
            rel_regions.append([])

    best_region = None
    best_key = (0, 0)
    for i, subsegments in enumerate(rel_regions):
        if not subsegments:
            continue
        first = subsegments[0]
        last = subsegments[-1]
        n_windows = last.end_ix - first.start_ix
        if n_windows < min_reliable_windows:
            continue

        region = Interval(first.dupl_region.region1.chrom_id, first.dupl_region.region1.start,
            last.dupl_region.region1.end)
        key = (n_windows, len(region))
        if key > best_key:
            best_region = region
            best_key = key
    return best_region
