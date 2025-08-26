#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author: KMC20
# Date: 2025/7/12
# Function: Predict DRIM-ANN's performance and record it in excels and diagrams.

import argparse
from xlwt import Workbook
from re import search
from copy import deepcopy
from ast import literal_eval
from math import ceil, log
from os.path import join
from matplotlib import pyplot as plt


C_STRUCT_ALIGNBYTES = 4
BitWidth = {
    'c': 1,
    'q': 1,
    'r': 2,
    'cb': 1,
    'a': 1,
    'sqt': 2,
    'l': 4,
    'd': 4
}  # Unit: Byte
RANDOM_MEM_COEFFICIENT = 484.051 / 112.368

def performance_model(N, D, K, P, C, M, CB, Q, P_dpu, F, PE, BW, BW_device):
    '''
    Estimate performance based on input configures.
    :param N:           The size of the base dataset.
    :param D:           The dimension of queries/centroids.
    :param K:           The amount of neighbors.
    :param P:           The amount of fetched clusters.
    :param C:           The average size of clusters.
    :param M:           The slice amount of vectors.
    :param CB:          The amount of codebook entries.
    :param Q:           The size of query batches.
    :param P_dpu:       The average amount of fetched clusters on each DPU.
    :param F:           The frequency of the processor used by each phase.
    :param PE:          The amount of the processor (for CPU, the total amount of hyperthreads) used by each phase.
    :param BW:          The bandwidth of the memory chip used by each phase.
    :param BW_device:   The bandwidth of memory devices.
    :return:            Estimated latency.
    '''
    CP = {}
    IO = {}
    IO_W = {}
    IO_M = {}
    CP["CL"] = Q * ceil(N / C) * (3 * ceil(D * BitWidth["q"] * 4 * 8 / 256) - 1 + log(P) - 1)  # ceil(D * BitWidth["q"] * 4 * 8 / 256): AVX2 for vector processing
    IO["CL"] = Q * ceil(N / C) * ((BitWidth["c"] + BitWidth["q"]) * D + (BitWidth["q"] * 4 + BitWidth["a"]) * (log(P) + 1))  # Not struct, non-alignment
    CP["CL"] += Q * ceil(N / C) * (2 * ceil(D * BitWidth["q"] * 4 * 8 / 256)  + (log(P) + 1))  # Cache loading instructions
    P = P_dpu  # The following phases are completed on DPUs
    CP["RC"] = Q * P * D
    IO["RC"] = (BitWidth["c"] + BitWidth["q"]) * Q * P * D
    IO_W["RC"] = BitWidth["r"] * Q * P * D
    IO_M["RC"] = (BitWidth["c"] + BitWidth["q"]) * Q * P * D
    CP["LC"] = Q * P * CB * (D / M * 3 - 1) * M
    IO["LC"] = Q * P * CB * (BitWidth["q"] * 2 * D + BitWidth["l"] * M)
    IO_W["LC"] = Q * P * (CB * BitWidth["l"] * M + BitWidth["r"] * D * CB + CB * BitWidth["sqt"] * D)
    IO_M["LC"] = Q * P * CB * BitWidth["cb"] * D
    CP["DC"] = Q * P * C * (M - 1)
    IO["DC"] = Q * P * C * (M * (BitWidth["a"] + BitWidth["l"]) + BitWidth["l"])
    IO_W["DC"] = Q * P * C * (M * BitWidth["l"] + BitWidth["d"])
    IO_M["DC"] = Q * P * C * (M * BitWidth["a"])
    CP["TS"] = Q * P * C * (log(K) - 1)
    IO["TS"] = Q * P * C * (log(K) + 1) * (BitWidth["l"] + BitWidth["a"])
    IO_W["TS"] = Q * P * C * (log(K) + 1) * ceil((BitWidth["l"] + BitWidth["a"]) / C_STRUCT_ALIGNBYTES) * C_STRUCT_ALIGNBYTES  # C struct: 4-byte alignment on DPUs
    IO_M["TS"] = Q * K * ceil((BitWidth["l"] + BitWidth["a"]) / C_STRUCT_ALIGNBYTES) * C_STRUCT_ALIGNBYTES                     # C struct: 4-byte alignment on DPUs
    # Each WRAM loading also costs a cycle according to DPU traces. Appended below
    CP["RC"] += 3 * Q * P * D
    CP["LC"] += Q * P * CB * (M + 2 * D) * BitWidth["l"] / BitWidth["r"]
    CP["DC"] += Q * P * C * (M * 2 + 1)
    CP["TS"] += Q * P * C * (log(K) + 1) + Q * K
    lat = {}
    lat["CL"] = max(CP["CL"] / (F["CL"] * PE["CL"]), min(N * M * BitWidth["a"], IO["CL"]) / BW["CL"])
    lat["RC"] = max(CP["RC"] / (F["RC"] * PE["RC"]), IO_W["RC"] / BW_device["WRAM"] + IO_M["RC"] / BW_device["MRAM"])
    lat["LC"] = max(CP["LC"] / (F["LC"] * PE["LC"]), IO_W["LC"] / BW_device["WRAM"] + IO_M["LC"] / BW_device["MRAM"]) * 1.6  # Redundant load coefficient (base: 0.6 * 12 / 7 for 12 extra instructions per loop before loop tiling and address caching): 0.6 = 0.6 * 12 / 7 * 7 / 12; * 12 / 7: DC->LC; * 7 / 12: (12 - 5) / 12
    lat["DC"] = max(CP["DC"] / (F["DC"] * PE["DC"]), IO_W["DC"] / BW_device["WRAM"] + IO_M["DC"] / BW_device["MRAM"]) * 1.34  # Redundant load coefficient (base: 0.6 for 7 extra instructions per loop before loop tiling): 0.34 = 0.6 * (7 - 3) / 7
    lat["TS"] = max(CP["TS"] / (F["TS"] * PE["TS"]), IO_W["TS"] / BW_device["WRAM"] + IO_M["TS"] / BW_device["MRAM"])
    latency = max(lat["CL"], lat["RC"] + lat["LC"] + lat["DC"] + lat["TS"])

    return latency

def readLog(data, log_file_name, fixed_configs, tested_config, codebook_entry_amount, dpu_amount, F, PE, BW, BW_device) -> None:
    '''
    Read logs from file for visualization.
    :param data:                    The dict for data in the output diagram. The key represents x axis, and the value represents y axis. Note that the type of the key must be int while the value is float.
    :param log_file_name:           The full name of the input log file.
    :param fixed_configs:           The config(s) to be fixed during the search. Note that the type of the value must be int.
    :param tested_config:           The config to be tested.
    :param codebook_entry_amount:   The amount of codebook entries.
    :param dpu_amount:              The amount of PIM-DPUs.
    :param F:                       The frequency of the processor used by each phase.
    :param PE:                      The amount of the processor (for CPU, the total amount of hyperthreads) used by each phase.
    :param BW:                      The bandwidth of the memory chip used by each phase.
    :param BW_device:               The bandwidth of memory devices.
    :return:                        None.
    '''
    pat = {
        "nlist": "clustersFileName = .*C(.*)M.*.clusters,",
        "nprobe": "nprobe = (.*), DPUGroupSize",
        "dataset": "clustersFileName =.*[^a-zA-Z0-9]([a-zA-Z0-9]*)C.*.clusters,",
        "N": "clustersFileName =.*[^a-zA-Z0-9][a-zA-Z]*([a-zA-Z0-9]*)C.*.clusters,",
        "dim": "dimAmt = (.*), neighborAmt",
        "K": "neighborAmt = (.*), sliceAmt",
        "M": "sliceAmt = (.*), queryBatchSize",
        "query_batch_size": "queryBatchSize = (.*), nprobe",
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
                N = search(pat["N"], log_row).groups()[0]
                if N[-1].lower() < 'a' or N[-1].lower() > 'z':
                    N = int(N)
                else:
                    Nunit = N[-1].lower()
                    if Nunit == 'k':
                        N = int(N[:-1]) * 1000
                    elif Nunit == 'm':
                        N = int(N[:-1]) * 1000000
                    elif Nunit == 'b':
                        N = int(N[:-1]) * 1000000000
                dim = int(search(pat["dim"], log_row).groups()[0])
                K = int(search(pat["K"], log_row).groups()[0])
                M = int(search(pat["M"], log_row).groups()[0])
                query_batch_size = int(search(pat["query_batch_size"], log_row).groups()[0])
                nprobe = int(search(pat["nprobe"], log_row).groups()[0])
                nlist = int(search(pat["nlist"], log_row).groups()[0])

                data_key = int(search(pat[tested_config], log_row).groups()[0])
                log_row = log_file.readline()
                while log_row:
                    obj = search(pat_perf, log_row)
                    if obj is not None:
                        data[data_key] = performance_model(N, dim, K, nprobe, N / nlist, M, codebook_entry_amount, query_batch_size, ceil(nprobe * query_batch_size / dpu_amount) * dpu_amount / query_batch_size, F, PE, BW, BW_device)
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
    parser.add_argument('--log-file-name', type=str, default='output.log', help='The full name, i.e. path + name, of the input log file (default: output.log)')
    parser.add_argument('--excel-dir-name', type=str, default='Excels/', help='The path name, of the output result excel workbook based on the data in the input log file (default: Excels/)')
    parser.add_argument('--diagram-dir-name', type=str, default='Figures/', help='The path name, of the output result diagram based on the data in the input log file (default: Figures/)')
    # Constant params
    parser.add_argument('--codebook-entry-amount', type=int, default='256', help='The amount of codebook entries. Used by the performance model (default: 256)')
    parser.add_argument('--dpu-amount', type=int, default='2560', help='The amount of PIM-DPUs. Used by the performance model (default: 2560)')
    parser.add_argument('--host-thread-amount', type=int, default='64', help='The amount of host threads. Used by the performance model (default: 64)')
    parser.add_argument('--query-amount', type=int, default='10000', help='The amount of queries. Used for convertion from latency to throughput (default: 10000)')
    args = parser.parse_args()

    # Check for configure params.
    if args.codebook_entry_amount < 1:
        raise ValueError(f'Argument `--codebook-entry-amount` should be positive! The input `{args.codebook_entry_amount}` is illegal!')
    if args.dpu_amount < 1:
        raise ValueError(f'Argument `--dpu-amount` should be positive! The input `{args.dpu_amount}` is illegal!')
    if args.query_amount < 1:
        raise ValueError(f'Argument `--query-amount` should be positive! The input `{args.query_amount}` is illegal!')

    # Record configure params.
    for arg_key in parser._actions[1:]:
        arg_key = arg_key.dest
        print(('{}: {}').format(arg_key, eval('args.{}'.format(arg_key))))

    # Hardware params.
    F = {
        # 'CL': 2100_000_000,
        'CL': 2400_000_000,  # Intel Xeon Silver 4210R
        'RC': 450_000_000,
        'LC': 450_000_000,
        'DC': 450_000_000,
        'TS': 450_000_000
    }  # Unit: Hz
    PE = {
        'CL': args.host_thread_amount,
        'RC': args.dpu_amount,
        'LC': args.dpu_amount,
        'DC': args.dpu_amount,
        'TS': args.dpu_amount
    }
    BW = {
        'CL': 19.2 * 1024 * 1024 * 1024,
        'RC': args.dpu_amount * 1024 * 1024 * 1024,
        'LC': args.dpu_amount * 1024 * 1024 * 1024,
        'DC': args.dpu_amount * 1024 * 1024 * 1024,
        'TS': args.dpu_amount * 1024 * 1024 * 1024
    }  # Unit: B/s
    BW_device = {
        'DDR4': 19.2 * 1024 * 1024 * 1024,
        'WRAM': args.dpu_amount * 1612.56 * 1024 * 1024 / RANDOM_MEM_COEFFICIENT,  # Random accesses take the major part. The coeffcient comes from reproduced official benchmark
        'MRAM': args.dpu_amount * 573.79 * 1024 * 1024
    }  # Unit: B/s
    
    fixed_config = {"nprobe": "96", "dataset": "sift100m"}
    tested_config = "nlist"
    fig_data = {}
    readLog(fig_data, args.log_file_name, fixed_config, tested_config, args.codebook_entry_amount, args.dpu_amount, F, PE, BW, BW_device)
    for fig_data_key in fig_data:  # Convert latency results to throughput
        fig_data[fig_data_key] = args.query_amount / fig_data[fig_data_key]
    write_sheet(args.excel_dir_name, fixed_config, tested_config, fig_data)
    draw_diagram(args.diagram_dir_name, fixed_config, tested_config, fig_data)
    
    fixed_config = {"nlist": "16384", "dataset": "sift100m"}
    tested_config = "nprobe"
    fig_data = {}
    readLog(fig_data, args.log_file_name, fixed_config, tested_config, args.codebook_entry_amount, args.dpu_amount, F, PE, BW, BW_device)
    for fig_data_key in fig_data:  # Convert latency results to throughput
        fig_data[fig_data_key] = args.query_amount / fig_data[fig_data_key]
    write_sheet(args.excel_dir_name, fixed_config, tested_config, fig_data)
    draw_diagram(args.diagram_dir_name, fixed_config, tested_config, fig_data)
    
    fixed_config = {"nprobe": "96", "dataset": "deep100m"}
    tested_config = "nlist"
    fig_data = {}
    readLog(fig_data, args.log_file_name, fixed_config, tested_config, args.codebook_entry_amount, args.dpu_amount, F, PE, BW, BW_device)
    for fig_data_key in fig_data:  # Convert latency results to throughput
        fig_data[fig_data_key] = args.query_amount / fig_data[fig_data_key]
    write_sheet(args.excel_dir_name, fixed_config, tested_config, fig_data)
    draw_diagram(args.diagram_dir_name, fixed_config, tested_config, fig_data)
    
    fixed_config = {"nlist": "16384", "dataset": "deep100m"}
    tested_config = "nprobe"
    fig_data = {}
    readLog(fig_data, args.log_file_name, fixed_config, tested_config, args.codebook_entry_amount, args.dpu_amount, F, PE, BW, BW_device)
    for fig_data_key in fig_data:  # Convert latency results to throughput
        fig_data[fig_data_key] = args.query_amount / fig_data[fig_data_key]
    write_sheet(args.excel_dir_name, fixed_config, tested_config, fig_data)
    draw_diagram(args.diagram_dir_name, fixed_config, tested_config, fig_data)
