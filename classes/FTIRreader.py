# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:33:54 2026

@author: justas
"""

import brukeropus
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



class FTIRreader:
    """
    A class to process and visualize polarization-dependent FTIR data from OPUS files.

    This class automates the reading of binary OPUS files, extracts metadata from filenames,
    interpolates data onto a common wavenumber axis, calculates mean reflectance, 
    and propagates standard error of the mean (SEM).

    Attributes:
        folder (str): Path to the directory containing the OPUS files.
        sample_prefix (str): The filename prefix identifying sample measurements.
        reference_prefix (str): The filename prefix identifying reference (e.g., Gold) measurements.
        x_limits (tuple): The default (min, max) wavenumber range for plotting.
        df (pd.DataFrame): Master dataframe containing filenames, metadata, and raw counts.
        merged (pd.DataFrame): Processed dataframe indexed by polarization angle, 
            containing averaged reflectance and propagated SEM.
        common_x (np.ndarray): The master wavenumber axis used for interpolation.
    """
    
    def __init__(self, folder, sample_prefix, reference_prefix, x_limits=(440, 1500)):
        """
        Initializes the FTIRreader and automatically triggers the data processing methods.

        Args:
            folder (str): Directory containing the OPUS files.
            sample_prefix (str): Prefix used for sample files (e.g., 'BlueBronze').
            reference_prefix (str): Prefix used for reference files (e.g., 'GoldMirror').
            x_limits (tuple, optional): Plotting x-axis limits. Defaults to entered values.
        """
        self.folder = folder
        self.sample_prefix = sample_prefix
        self.reference_prefix = reference_prefix
        self.x_limits = x_limits
        
        
        self._get_filenames()
        self._get_dataframe()
        self._read_data()
        self._get_mean_and_sem()
        
        
    
    def _get_filenames(self):
        """Scans the source folder and stores all filenames in a list."""
        self.filenames = os.listdir(self.folder)
    
    
    def _get_dataframe(self):
        """
        Creates a pandas DataFrame and extracts metadata using regex.
        
        Parsed metadata includes:
            - Sample name
            - Polarization angle (pol_angle)
            - Run number
        """
        self.df = pd.DataFrame({"filename": self.filenames})
        pattern = r'(?P<sample>^.*?)-Pol(?P<pol_angle>\d+)deg-Run(?P<run>[\d.]+)'
        metadata = self.df['filename'].str.extract(pattern)
        self.df = pd.concat([self.df, metadata], axis=1)
    
    
    def _read_data(self):
        """
        Reads OPUS files, establishes a common x-axis, and interpolates counts to match the commonx-axis.
        
        The first file in the directory defines the common wavenumber (x) axis.
        All subsequent files are interpolated onto this axis to ensure mathematical 
        compatibility for averaging and calculating reflectance.
        """
        
        # Read the contents of the 1st file and set its x axis to be the common one for the rest of the files
        first_file_path = os.path.join(self.folder, self.df['filename'].iloc[0])
        data_0 = brukeropus.read_opus(first_file_path)
        self.common_x = data_0.sm.x
        if self.common_x[0] > self.common_x[-1]:
            self.common_x = self.common_x[::-1]
        
        del data_0
        
        num_files = len(self.df)
        counts = []
        
        for i, row in self.df.iterrows():
            path = os.path.join(self.folder, row["filename"])
            
            try:
                data = brukeropus.read_opus(path)
                x, y = data.sm.x, data.sm.y
                
                # Check if the x data is descending - important for interpolation
                if x[0] > x[-1]:
                    x, y = x[::-1], y[::-1]
                
                counts.append(np.interp(self.common_x, x, y))
                
                if i % 10 == 0:
                    print(f"Processed file {i}/{num_files}")
                elif i == len(self.df) - 1:
                    print(f"Processed file {i+1}/{num_files}")
                    print("Finished processing OPUS files")
                
                del data
            
            except Exception as e:
                print(f"Error reading {row['filename']}: {e}")
        self.df["counts"] = counts
        print("Counts loaded into dataframe df")
        # print(self.df)
        
        
    def _get_mean_and_sem(self):
        """
        Groups data by sample/polarization to calculate averages and uncertainty.
        
        Calculates:
            1. Mean and SEM for raw counts.
            2. Reflectance (Sample Mean / Reference Mean).
            3. Propagated SEM for reflectance using relative uncertainty quadrature.
        """
        stats = self.df.groupby(["sample", "pol_angle"])["counts"].agg(
            mean_y = lambda x: np.mean(np.vstack(x), axis=0),
            std_y  = lambda x: np.std(np.vstack(x), axis=0),
            sem_y = lambda x: np.std(np.vstack(x), axis=0) / np.sqrt(len(x)),
            ).reset_index()
        
        # Group sample and reference counts
        references = stats[stats["sample"] == self.reference_prefix].set_index("pol_angle")
        samples = stats[stats["sample"] == self.sample_prefix].set_index("pol_angle")
        
        self.merged = samples.merge(references, on="pol_angle", suffixes=("_s", "_ref"))
        
        self.merged["reflectance"] = self.merged["mean_y_s"] / self.merged["mean_y_ref"]
        
        # Propagate the SEM
        self.merged["sem_reflectance"] = self.merged.apply(
            lambda row: row["reflectance"] * np.sqrt(
                (row["sem_y_s"] / row["mean_y_s"])**2 + 
                (row["sem_y_ref"] / row["mean_y_ref"])**2
                ),
                axis = 1
            )
        # print(self.merged)
    
    
    
    def plot_R_at_pol_angle(self, pol_angle):
        """
        Generates a two-panel plot for a given polarization angle.

        Args:
            pol_angle (int or str): The polarization angle to plot the reflectance at.

        The top panel displays the calculated reflectance with a shaded SEM envelope.
        The bottom panel shows the raw average sample and reference counts.
        """
        
        try:
            row = self.merged.loc[str(pol_angle)]
        except KeyError:
            print(f"Polarization angle {pol_angle} was not found.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})
        fig.suptitle(f"FTIR: pol. angle={pol_angle} deg")
        
        refl = row["reflectance"]
        refl_sem = row["sem_reflectance"]
        sem_s = row["sem_y_s"]
        sem_ref = row["sem_y_ref"]
        
        ax1.plot(self.common_x, refl, color='black', label='Reflectance')
        ax1.fill_between(self.common_x, refl - refl_sem, refl + refl_sem, color='gray', alpha=0.3, label='Propagated SEM')
        ax1.tick_params(axis='x', which='both', top=True, labeltop=True)  # enables the display of wavenumber values at the top of the 1st plot
        ax1.axhline(y=0, color='k', linestyle='--')
        ax1.axhline(y=1, color='k', linestyle='--')
        
        ax2.plot(self.common_x, row['mean_y_s'], label='Sample Avg', color='blue')
        ax2.fill_between(self.common_x, row['mean_y_s'] - sem_s, row['mean_y_s'] + sem_s, color='blue', alpha=0.3)
        
        ax2.plot(self.common_x, row['mean_y_ref'], label='Reference Avg', color='red')
        ax2.fill_between(self.common_x, row['mean_y_ref'] - sem_ref, row['mean_y_ref'] + sem_ref, color='red', alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--')
        
        
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_xlim(self.x_limits[0], self.x_limits[1])
        ax1.set_ylim(0.15, 0.55)
        
        
        ax1.set_ylabel("$R$")
        ax2.set_ylabel("Counts (arb. units)")
        ax2.set_xlabel("Wavenumber (cm$^{-1}$)")
        
        
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        
    























