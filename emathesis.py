from nptdms import TdmsFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import kurtosis, skew
from tqdm import tqdm
import configparser
import logging
from MeasuredDataFile import MeaseredDataFile
import os


PROJECT_PATH = "/home/milan/Desktop/ema/emadataconverter/"
SOURCE_SUBFOLDER = "2020_12_18_NO_DEFECT/"
OUTPUT_SUBFOLDER = "2020_12_18_NO_DEFECT_OUTPUT/"
MERGE_FRAMES = True

FILENAME_TEMPLATE = "20201218_151015_?_SIN_30N_NO_DEFF_MAT.tdms"
CONGIF_PATH = "measured_data.conf"

#logging
log = logging.getLogger("emaconv")
log.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('/tmp/emadataconverter.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
#log.addHandler(fh)
log.addHandler(ch)


def load_variable(variable):
    filename = FILENAME_TEMPLATE.replace("?", variable)
    df_temp = TdmsFile(PROJECT_PATH + SOURCE_SUBFOLDER + filename).as_dataframe()
    return df_temp


def compute_factors(df_new, df_raw, n_final, rms=False, cf=False, mean=False, variance=False, kurt=False, skewness=False):
    name_arr = ["rms" if rms else None,
                "cf" if cf else None,
                "mean" if mean else None,
                "var" if variance else None,
                "kurt" if kurt else None,
                "skew" if skewness else None]
    name_arr = [item for item in name_arr if item is not None]

    for column in df_raw.columns:
        for factor_name in name_arr:
            df_new[column + "_" + factor_name] = np.nan

    num_colums = len(df_raw.columns)
    frame_len = df_raw.shape[0]
    window_dec, window_len = math.modf(frame_len / n_final)
    window_len = int(window_len)

    left_samples = 0.0
    i = 0

    for num_row in tqdm(range(n_final)):
        left_samples = left_samples + window_dec if left_samples < 1 else left_samples + window_dec - 1
        n = window_len if left_samples < 1 else window_len + 1

        row = []
        for column in df_raw.columns:
            window = df_raw.loc[i:i + n, [column]].to_numpy()
            # print(window)

            if rms:
                rms = np.sqrt(np.mean(window ** 2))
                row.append(rms)
            if cf:
                cf = abs(max(window))[0] / rms
                row.append(cf)
            if mean:
                mean = np.mean(window)
                row.append(mean)
            if variance:
                variance = np.var(window, ddof=1)
                row.append(variance)
            if kurt:
                kurt = kurtosis(window)[0]
                row.append(kurt)
            if skewness:
                skewness = skew(window)[0]
                row.append(skewness)

        df_new.loc[len(df_new)] = row
        i += n

    print(i, df_raw.shape[0])
    # return df_new


def upsample(df, period, period_resampled):
    df = df.set_index(pd.date_range('1/1/2000', periods=df.shape[0], freq=period))
    df = df.resample(period_resampled).interpolate(method="linear")
    df = df.reset_index()
    df = df.drop("index", 1)
    return df


def get_files_list():
    files_list = [f for f in os.listdir(PROJECT_PATH + SOURCE_SUBFOLDER) if f.endswith('.tdms')]
    for i in range(len(files_list)):
        tmp_list = list(files_list[i])
        tmp_list[16] = "?"
        files_list[i] = ''.join(tmp_list)

    return set(files_list)


def save_dataframe(df, filename, hdf=True, csv=False):
    # save to csv
    if csv:
        filename = FILENAME_TEMPLATE.replace("?", "all").replace("tdms", "csv")
        df.to_csv(PROJECT_PATH + OUTPUT_SUBFOLDER + filename, index=False)

    # save to hdf
    if hdf:
        filename = filename.replace("?", "all").replace("tdms", "h5")
        df.to_hdf(PROJECT_PATH + OUTPUT_SUBFOLDER + filename, key="df", mode="w", index=False)


def create_dataset(files_list, config):
    df_merge = pd.DataFrame()

    # main loop for creating new datasets
    for filename in files_list:
        print("INFO: Dataset creation for filename: {}".format(filename))
        file_path = PROJECT_PATH + SOURCE_SUBFOLDER + filename

        # slow data
        s_data = MeaseredDataFile(file_path, "s", config)
        df_length = s_data.data_raw.shape[0]
        log.info("Final length of this measurement is {}".format(df_length))

        # vibrations data
        v_data = MeaseredDataFile(file_path, "v", config)
        compute_factors(v_data.data_converted, v_data.data_raw, df_length, rms=True, cf=True, mean=False, variance=True,
                        kurt=True, skewness=False)

        # fast data
        f_data = MeaseredDataFile(file_path, "f", config)
        compute_factors(f_data.data_converted, f_data.data_raw, df_length, rms=True, cf=True, mean=True, variance=True,
                        kurt=True, skewness=True)

        # temperature data
        t_data = MeaseredDataFile(file_path, "t", config)
        t_data.data_converted = t_data.data_raw.copy()
        t_data.data_converted = upsample(t_data.data_converted, period="200ms", period_resampled="1ms")

        df = pd.concat([s_data.data_raw, v_data.data_converted, f_data.data_converted, t_data.data_converted], axis=1)
        min_len = min([s_data.data_raw.shape[0], v_data.data_converted.shape[0], f_data.data_converted.shape[0],
                       t_data.data_converted.shape[0]])
        print(s_data.data_raw.shape, v_data.data_converted.shape, f_data.data_converted.shape,
              t_data.data_converted.shape)
        df = df[:min_len]
        print(df.shape)
        log.info("INFO: Dataset for {} was successfully created with size {}".format(filename.replace("tdms", "h5"),
                                                                                     df.shape))
        save_dataframe(df, filename, hdf=True)

        if MERGE_FRAMES == True:
            df_merge = df_merge.append(df, ignore_index=True)

    save_dataframe(df_merge, "merged_no_deff.tdms", hdf=True)


def cleaned_dataset(files_list, config):
    # main loop for creating new datasets
    for filename in files_list:
        print("INFO: Dataset creation for filename: {}".format(filename))
        file_path = PROJECT_PATH + SOURCE_SUBFOLDER + filename

        # slow data
        s_data = MeaseredDataFile(file_path, "s", config)
        save_to_csv(s_data.data_raw, "s", filename)

        # vibrations data
        v_data = MeaseredDataFile(file_path, "v", config)
        save_to_csv(v_data.data_raw, "v", filename)

        # fast data
        f_data = MeaseredDataFile(file_path, "f", config)
        save_to_csv(f_data.data_raw, "f", filename)

        # temperature data
        t_data = MeaseredDataFile(file_path, "t", config)
        save_to_csv(t_data.data_raw, "t", filename)


def save_to_csv(df, variable, filename):
    filename = filename.replace("?", variable).replace("tdms", "csv")
    df.to_csv(PROJECT_PATH + OUTPUT_SUBFOLDER + filename, index=False)


def describe_dataset(files_list, config):
    filename = list(files_list)[0]

    print("INFO: Creating description and plots for filename: {}".format(filename))
    file_path = PROJECT_PATH + SOURCE_SUBFOLDER + filename
    data_obj = MeaseredDataFile(file_path, "s", config)
    print(filename)

    for feature in data_obj.data_converted.columns:
        print(feature)
        plt.plot(data_obj.data_converted[feature])
        plt.title(feature + " graph")
        plt.show()


if __name__ == '__main__':
    log.info("Data conversion started")
    config = configparser.ConfigParser()

    try:
        config.read(CONGIF_PATH)
        log.info("Config file: %s", CONGIF_PATH)
    except:
        log.error("Config not found in '%s'", CONGIF_PATH)
        exit(1)

    # get unique measurements filenames from a folder
    files_list = get_files_list()

    # funtion for creating whole dataset from more measurements
    # create_dataset(files_list, config)

    # describe_dataset(files_list, config)

    #create cleaned csv
    cleaned_dataset(files_list, config)