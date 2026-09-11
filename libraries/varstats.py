""" This class of functions is primarily for calculating statistics for light curves"""
import numpy as np
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit


class Varstats:

    @staticmethod
    def stetson_j_peak_and_cutoff(j_values, sigma_method="mirror_std", n_sigma=3.0):
        """
        Full pipeline: bin the J distribution, find its peak, estimate a robust
        core sigma, and return a 3-sigma (or n-sigma) variability cutoff.

        sigma_method: "mirror_std", "mad", or "gaussian_fit"

        Returns a dict with peak, sigma, cutoff, and the variable-candidate mask.
        """
        j_values = np.asarray(j_values)
        j_values = j_values[(~np.isnan(j_values)) & (j_values > 0)]

        peak_info = Varstats.find_peak(j_values)
        peak = peak_info["peak_value"]

        # sigma = Varstats.robust_core_sigma(j_values, peak, side="left", method=sigma_method)
        peak, sigma, _ = Varstats.gaussian_core_fit(j_values)

        cutoff = peak + n_sigma * sigma
        candidate_mask = j_values > cutoff

        return {
            "peak": peak,
            "sigma": sigma,
            "cutoff": cutoff,
            "n_sigma": n_sigma,
            "n_candidates": int(candidate_mask.sum()),
            "candidate_mask": candidate_mask,
            "bin_edges": peak_info["bin_edges"],
            "counts": peak_info["counts"],
        }

    @staticmethod
    def optimal_bins(data, method="fd"):
        """Freedman-Diaconis (default), Scott, Sturges, or a fixed int bin count."""
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        n = data.size
        if n < 2:
            raise ValueError("Need at least 2 data points to bin.")

        data_range = data.max() - data.min()
        if data_range == 0:
            return np.array([data.min() - 0.5, data.min() + 0.5]), 1

        if isinstance(method, int):
            n_bins = method
        elif method == "sturges":
            n_bins = int(np.ceil(np.log2(n) + 1))
        elif method == "scott":
            bin_width = 3.5 * data.std(ddof=1) / (n ** (1 / 3))
            n_bins = max(1, int(np.ceil(data_range / bin_width)))
        else:  # Freedman-Diaconis
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                n_bins = int(np.ceil(np.log2(n) + 1))
            else:
                bin_width = 2 * iqr / (n ** (1 / 3))
                n_bins = max(1, int(np.ceil(data_range / bin_width)))

        n_bins = max(1, n_bins)
        bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)
        return bin_edges, n_bins

    @staticmethod
    def find_peak(data, bin_edges=None, method="fd", interpolate=True):
        """Bin `data` and return the histogram's peak (mode), with optional
        parabolic sub-bin interpolation."""
        data = np.asarray(data)
        data = data[~np.isnan(data)]

        if bin_edges is None:
            bin_edges, _ = Varstats.optimal_bins(data, method=method)

        counts, edges = np.histogram(data, bins=bin_edges)
        centers = (edges[:-1] + edges[1:]) / 2

        peak_idx = int(np.argmax(counts))
        peak_value = centers[peak_idx]

        if interpolate and 0 < peak_idx < len(counts) - 1:
            y0, y1, y2 = counts[peak_idx - 1], counts[peak_idx], counts[peak_idx + 1]
            denom = (y0 - 2 * y1 + y2)
            if denom != 0:
                offset = np.clip(0.5 * (y0 - y2) / denom, -1, 1)
                bin_width = centers[1] - centers[0] if len(centers) > 1 else (edges[1] - edges[0])
                peak_value = centers[peak_idx] + offset * bin_width

        return {
            "peak_value": peak_value,
            "peak_count": int(counts[peak_idx]),
            "bin_edges": edges,
            "counts": counts,
            "centers": centers,
        }

    def gaussian_core_fit(data, bin_edges=None, method="fd", fit_window_sigma=2.5):
        """
        Alternative to robust_core_sigma: fit a Gaussian directly to the
        histogram core using scipy curve_fit. Iterates once to restrict the fit
        window to points near the peak, so the high-tail doesn't bias the fit.

        Returns (peak, sigma, amplitude) from the Gaussian fit.
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]

        if bin_edges is None:
            bin_edges, _ = Varstats.optimal_bins(data, method=method)
        counts, edges = np.histogram(data, bins=bin_edges)
        centers = (edges[:-1] + edges[1:]) / 2

        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # initial guess: use histogram peak + a rough std for a first pass
        p0_mu = centers[np.argmax(counts)]
        p0_sigma = data.std() / 2 if data.std() > 0 else 1.0
        p0_amp = counts.max()

        try:
            popt, _ = curve_fit(gaussian, centers, counts,
                                p0=[p0_amp, p0_mu, p0_sigma], maxfev=10000)
            amp, mu, sigma = popt
            sigma = abs(sigma)

            # second pass: restrict to a window around the fitted peak to reduce
            # tail contamination, then refit
            mask = np.abs(centers - mu) <= fit_window_sigma * sigma
            if mask.sum() >= 4:
                popt2, _ = curve_fit(gaussian, centers[mask], counts[mask],
                                     p0=[amp, mu, sigma], maxfev=10000)
                amp, mu, sigma = popt2
                sigma = abs(sigma)

            return mu, sigma, amp
        except RuntimeError:
            raise RuntimeError("Gaussian fit failed to converge; try robust_core_sigma() instead.")

    @staticmethod
    def robust_core_sigma(data, peak, side="left", method="mirror_std"):
        """
        Estimate the width (sigma) of the core distribution around `peak`,
        ignoring contamination from a one-sided tail (e.g. real variables
        pulling J to high values).

        side: "left" assumes the tail of interest is on the high side, so only
              data <= peak is trusted and mirrored to estimate sigma. Use
              "right" if your contamination is on the low side instead.

        method:
            "mirror_std" - reflect the trusted side about the peak and take the
                            std of the combined (symmetrized) sample. Simple and
                            robust to a one-sided tail.
            "mad"        - median absolute deviation of the trusted side about
                            the peak, scaled by 1.4826 to be a consistent
                            estimator of sigma for a Gaussian core.
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]

        if side == "left":
            trusted = data[data <= peak]
        else:
            trusted = data[data >= peak]

        if trusted.size < 5:
            raise ValueError("Not enough points on the trusted side to estimate sigma; "
                             "check that `peak` is reasonable and you have enough data.")

        if method == "mad":
            residuals = np.abs(trusted - peak)
            mad = np.median(residuals)
            sigma = 1.4826 * mad
        else:  # mirror_std
            residuals = trusted - peak  # all <= 0 (or >= 0 for "right")
            mirrored = np.concatenate([residuals, -residuals])
            sigma = mirrored.std(ddof=1)

        return sigma

    @staticmethod
    def stetson_metrics(full_mag, full_err):
        """ This function calculates the J, L, & K Stetson metrics. This based on the code used for the Oelkers+2018
        KELT variable catalog calculations.

        :parameter full_mag: A numpy array with the magnitude values, not sigma-clipped
        :parameter full_err: A numpy array with the magnitude errors, not sigma-clipped

        :return j, k, l - the stetson index values are returned
        """

        if (len(full_mag) > 0) & (len(full_err) > 0):
            # do the sigma clipping
            clipped_mag = sigma_clip(full_mag, sigma=3, maxiters=5)
            valid_mask = ~clipped_mag.mask

            # clip the mag and err
            mag = full_mag[valid_mask]
            err = full_err[valid_mask]

            # set up a few variables
            w_k = 1.0  # Weighting Factor, set to 1 to not ignore flares or ebs
            mean_mag = np.mean(mag)  # mean magnitude
            num_pts = len(mag)

            j_top = np.zeros(num_pts)
            j_btm = np.zeros(num_pts)
            k_top = np.zeros(num_pts)
            k_btm = np.zeros(num_pts)

            for idx in np.arange(0, num_pts - 2, 2):

                sgn_i = ((mag[idx] - mean_mag) / err[idx]) * np.sqrt((num_pts / (num_pts - 1)))
                sgn_j = ((mag[idx + 1] - mean_mag) / err[idx + 1]) * np.sqrt((num_pts / (num_pts - 1)))

                p_k = sgn_i * sgn_j  # pg 853 Stetson 1996
                if p_k > 0.0:
                    sgn_pk = 1.0
                if p_k == 0.0:
                    sgn_pk = 0.0
                if p_k < 0.0:
                    sgn_pk = -1.0

                j_top[idx] = w_k * sgn_pk * (np.sqrt(np.abs(p_k)))  # Kinemuchi eq.1 (Numerator)
                j_btm[idx] = w_k  # Kinemuchi eq.1 (Denominator)
                k_top[idx] = np.abs(sgn_i)  # Kinemuchi eq.5 (Numerator)
                k_btm[idx] = np.abs(sgn_i ** 2.0)  # Kinemuchi eq.5 (Denominator)

            j = np.sum(j_top) / np.sum(j_btm)  # Stetson J

            # Stetson K
            if np.sum(k_btm) != 0:
                k = ((1.0 / num_pts) * np.sum(k_top)) / (np.sqrt((1.0 / num_pts) * np.sum(k_btm)))  # Stetson K
            else:
                k = 0.

            # Stetson L
            l = (j * k) / 0.7908

            return j, k, l
        else:
            return -9.999, -9.999, -9.999