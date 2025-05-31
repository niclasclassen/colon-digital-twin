import SimpleITK as sitk
import numpy as np
import csv


def load_mha_file(file_path):
    """
    Loads a .mha file and returns its content
    :param file_path: path to the file
    :return: the content of the file
    """
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image)


def save_as_mha_file(image, file_path):
    """
    Saves a numpy array as a .mha file
    :param image: numpy array to save
    :param file_path: path to the file
    """
    sitk.WriteImage(sitk.GetImageFromArray(image), file_path)


def create_log_file(log_file_name, header):
    """
    Creates a log file with a header
    :param log_file_name: name of the log file
    :param header: header of the log file
    """
    with open(log_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
