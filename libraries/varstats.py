""" This class of functions is primarily for calculating statistics for light curves"""
import numpy as np
import pandas as pd


class Varstats:

    @staticmethod
    def stetson_metrics(mag, err):
        """ This function calculates the J, L, & K Stetson metrics. This based on the code used for the Oelkers+2018
        KELT variable catalog calculations.

        :parameter mag: A numpy array with the magnitude values
        :parameter err: A numpy array with the magnitude errors

        :return j, k, l - the stetson index values are returned
        """

        # set up a few variables
        w_k = 1.0  # Weighting Factor, set to 1 to not ignore flares or ebs
        mean_mag = np.mean(mag)  # mean magnitude
        num_pts = len(mag)

        j_top = np.zeros(num_pts)
        j_btm = np.zeros(num_pts)
        k_top = np.zeros(num_pts)
        k_btm = np.zeros(num_pts)

        for idx in np.arange(0, num_pts - 2, 2):

            sgn_i = (mag[idx] - mean_mag) / (err[idx]) * (np.sqrt(num_pts / (num_pts - 1)))
            sgn_j = (mag[idx + 1] - mean_mag) / (err[idx + 1]) * (np.sqrt(num_pts / (num_pts - 1)))

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