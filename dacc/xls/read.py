

__author__ = 'thor'

import xlrd


def get_sheet_names(xls_filepath):
    return xlrd.open_workbook(xls_filepath, on_demand=True).sheet_names()
