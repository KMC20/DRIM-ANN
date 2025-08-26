#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author: KMC20
# Date: 2025/7/12
# Function: Convert DRIM-ANN's log to excels and diagrams.

import argparse
from xlwt import Workbook
from re import search
from copy import deepcopy
from ast import literal_eval
from math import ceil, log
from os.path import join
from matplotlib import pyplot as plt


def readLog(data, log_file_name, fixed_configs, tested_config) -> None:
    '''
    Read logs from file for visualization.
    :param data:            The dict for data in the output diagram. The key represents x axis, and the value represents y axis. Note that the type of the key must be int while the value is float.
    :param log_file_name:   The full name of the input log file.
    :param fixed_configs:   The config(s) to be fixed during the search. Note that the type of the value must be int.
    :param tested_config:   The config to be tested.
    :return:                None.
    '''
    pat = {
        "nlist": "clustersFileName = .*C(.*)M.*.clusters,",
        "nprobe": "nprobe = (.*), DPUGroupSize",
        "dataset": "clustersFileName =.*[^a-zA-Z0-9]([a-zA-Z0-9]*)C.*.clusters,",
    }
    pat_config = "Config: "
    pat_perf = "\[Host\]  Total time for query searching: (.*)s"

    # Load configures.
    with open(log_file_name, 'r') as log_file:
        log_row = log_file.readline()
        while log_row:
            is_fit = True
            for fixed_config in fixed_configs.items():  # Note: the value of tested_config should be and only be int!
                obj = search(pat[fixed_config[0]], log_row)
                if obj is None or obj.groups()[0] != fixed_config[1]:
                    is_fit = False
                    break
            if is_fit is True:
                data_key = int(search(pat[tested_config], log_row).groups()[0])
                log_row = log_file.readline()
                while log_row:
                    obj = search(pat_perf, log_row)
                    if obj is not None:
                        data[data_key] = float(obj.groups()[0])
                        log_row = log_file.readline()
                        break
                    obj = search(pat_config, log_row)
                    if obj is not None:  # Something wrong during DPU running so that no performance data are there for current config. The read config row is used for next detection
                        break
                    log_row = log_file.readline()
            else:
                log_row = log_file.readline()

def write_sheet(excel_dir_name, fixed_configs, tested_config, data) -> None:
    '''
    Convert log data to an excel.
    :param excel_dir_name:      The directory of the output excel file.
    :param fixed_configs:       The config(s) to be fixed during the search. Note that the type of the value must be int.
    :param tested_config:       The config to be tested.
    :param data:                The dict for data in the output diagram. The key represents x axis, and the value represents y axis. Note that the type of the key must be int while the value is float.
    :return:                    None.
    '''
    excel_file_name = tested_config
    for fixed_config in fixed_configs.items():
        excel_file_name = f"{excel_file_name}_{fixed_config[0]}{str(fixed_config[1])}"
    excel_file_name = join(excel_dir_name, f"{excel_file_name}.xls")

    result_excel = Workbook()
    sheet = result_excel.add_sheet(tested_config)
    tested_config_col_idx = len(fixed_configs)
    QPS_col_idx = tested_config_col_idx + 1
    for fixed_config_idx, fixed_config in enumerate(fixed_configs.items()):
        sheet.write(0, fixed_config_idx, label = fixed_config[0])
    sheet.write(0, tested_config_col_idx, label = tested_config)
    sheet.write(0, QPS_col_idx, label = 'Throughput (QPS)')
    result_idx = 1
    for result_idx, row in enumerate(data.items(), start = 1):
        for fixed_config_idx, fixed_config in enumerate(fixed_configs.items()):
            sheet.write(result_idx, fixed_config_idx, label = fixed_config[1])
        sheet.write(result_idx, tested_config_col_idx, label = row[0])
        sheet.write(result_idx, QPS_col_idx, label = row[1])
    result_excel.save(excel_file_name)

def draw_diagram(diagram_dir_name, fixed_configs, tested_config, data) -> None:
    '''
    Convert log data to a diagram.
    :param diagram_dir_name:    The directory of the output diagram file.
    :param fixed_configs:       The config(s) to be fixed during the search. Note that the type of the value must be int.
    :param tested_config:       The config to be tested.
    :param data:                The dict for data in the output diagram. The key represents x axis, and the value represents y axis. Note that the type of the key must be int while the value is float.
    :return:            Estimated latency.
    '''
    from logging import getLogger
    getLogger_disabled = getLogger('matplotlib.font_manager').disabled  # Avoid a huge amount of logging DEBUG outputs.
    getLogger('matplotlib.font_manager').disabled = True
    diagram_file_name = tested_config
    for fixed_config in fixed_configs.items():
        diagram_file_name = f"{diagram_file_name}_{fixed_config[0]}{str(fixed_config[1])}"
    diagram_file_name = join(diagram_dir_name, f"{diagram_file_name}.png")

    _, ax = plt.subplots()
    x, y = [], []
    for bucket in data.items():
        x.append(str(bucket[0]))
        y.append(bucket[1])
    ax.bar(x, y, color = 'orange', width = 0.3)
    ax.set_xlabel(tested_config)
    ax.set_ylabel('Throughput (QPS)')
    # ax.set_title(f'DRIM-ANN performance on UPMEM')
    # plt.legend()
    plt.savefig(diagram_file_name)
    getLogger('matplotlib.font_manager').disabled = getLogger_disabled  # Recover logger.


if __name__ == '__main__':
    # Get configure parameters.
    parser = argparse.ArgumentParser()
    # Disk file params
    parser.add_argument('--log-file-name', type=str, default='build/output.txt', help='The full name, i.e. path + name, of the input log file (default: output.log)')
    parser.add_argument('--excel-dir-name', type=str, default='Excels/', help='The path name, of the output result excel workbook based on the data in the input log file (default: Excels/)')
    parser.add_argument('--diagram-dir-name', type=str, default='Figures/', help='The path name, of the output result diagram based on the data in the input log file (default: Figures/)')
    # Constant params
    parser.add_argument('--query-amount', type=int, default='10000', help='The amount of queries. Used for convertion from latency to throughput (default: 10000)')
    args = parser.parse_args()

    # Check for configure params.
    if args.query_amount < 1:
        raise ValueError(f'Argument `--query-amount` should be positive! The input `{args.query_amount}` is illegal!')

    # Record configure params.
    for arg_key in parser._actions[1:]:
        arg_key = arg_key.dest
        print(('{}: {}').format(arg_key, eval('args.{}'.format(arg_key))))
    
    fixed_config = {"nprobe": "96", "dataset": "sift100m"}
    tested_config = "nlist"
    fig_data = {}
    readLog(fig_data, args.log_file_name, fixed_config, tested_config)
    for fig_data_key in fig_data:  # Convert latency results to throughput
        fig_data[fig_data_key] = args.query_amount / fig_data[fig_data_key]
    write_sheet(args.excel_dir_name, fixed_config, tested_config, fig_data)
    draw_diagram(args.diagram_dir_name, fixed_config, tested_config, fig_data)
    
    fixed_config = {"nlist": "16384", "dataset": "sift100m"}
    tested_config = "nprobe"
    fig_data = {}
    readLog(fig_data, args.log_file_name, fixed_config, tested_config)
    for fig_data_key in fig_data:  # Convert latency results to throughput
        fig_data[fig_data_key] = args.query_amount / fig_data[fig_data_key]
    write_sheet(args.excel_dir_name, fixed_config, tested_config, fig_data)
    draw_diagram(args.diagram_dir_name, fixed_config, tested_config, fig_data)
    
    fixed_config = {"nprobe": "96", "dataset": "deep100m"}
    tested_config = "nlist"
    fig_data = {}
    readLog(fig_data, args.log_file_name, fixed_config, tested_config)
    for fig_data_key in fig_data:  # Convert latency results to throughput
        fig_data[fig_data_key] = args.query_amount / fig_data[fig_data_key]
    write_sheet(args.excel_dir_name, fixed_config, tested_config, fig_data)
    draw_diagram(args.diagram_dir_name, fixed_config, tested_config, fig_data)
    
    fixed_config = {"nlist": "16384", "dataset": "deep100m"}
    tested_config = "nprobe"
    fig_data = {}
    readLog(fig_data, args.log_file_name, fixed_config, tested_config)
    for fig_data_key in fig_data:  # Convert latency results to throughput
        fig_data[fig_data_key] = args.query_amount / fig_data[fig_data_key]
    write_sheet(args.excel_dir_name, fixed_config, tested_config, fig_data)
    draw_diagram(args.diagram_dir_name, fixed_config, tested_config, fig_data)
