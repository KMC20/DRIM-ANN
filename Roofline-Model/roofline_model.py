#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author: KMC20
# Date: 2025/6/8
# Function: Roofline model of input datasets. The plot includes the roofline model of both input datasets and platforms

import argparse
from xlwt import Workbook
from math import log2
import json
from matplotlib import pyplot as plt


def roofline_model_datasets(**kwargs) -> dict:
    '''
    Estimate arithmetic intensity based on input configures.
    :param kwargs:   Configurations of datasets. Keys of 'dim', 'nlist', 'nprobe', 'M', 'query_batch_size', 'K', 'N' and 'CB' should be included
    :return:         Estimated arithmetic intensity.
    '''
    arithmetic_intensity = {}
    for name, config in kwargs.items():
        C_CL = config['query_batch_size'] *config['N'] / config['nlist'] * (3 * config['dim'] / 512 - 2 + log2(config['nprobe']))
        IO_CL = config['dim'] * config['query_batch_size'] * 4 + config['N'] / config['nlist'] * (config['dim'] * 8)
        C_RC = config['query_batch_size'] * config['nprobe'] * config['dim'] / 512
        IO_RC = (config['query_batch_size'] + config['nprobe'] ) * config['dim'] * 4
        C_LC = config['query_batch_size'] * config['nprobe'] * config['CB'] * config['dim'] * (3 * config['M'] - 1) / config['M'] / 512
        IO_LC = (4 + config['M'] * config['CB'] * 4)* config['query_batch_size'] * config['nprobe'] + config['dim'] * config['CB'] * 4
        C_DC = config['query_batch_size'] * config['nprobe'] * config['nlist'] * (config['M'] - 1) / 512
        IO_DC = min(config['N'] , config['query_batch_size'] * config['nprobe'] * (config['nlist'] * (config['M'] * 4 + 4) + config['M'] * config['CB'] * 4))
        C_TS = config['query_batch_size'] * config['nprobe'] * config['nlist'] * (log2(config['K']) - 1)
        IO_TS = config['query_batch_size'] * config['nprobe'] * (log2(config['K']) + 1) * 8
        C = C_CL + C_RC + C_LC + C_DC + C_TS
        IO = IO_CL + IO_RC + IO_LC + IO_DC + IO_TS

        arithmetic_intensity[name] = C / IO

    return arithmetic_intensity

def draw_plots(platforms_config, arithmetic_intensity, plot_file_name) -> None:
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
    from logging import getLogger
    getLogger_disabled = getLogger('matplotlib.font_manager').disabled  # Avoid a huge amount of logging DEBUG outputs.
    getLogger('matplotlib.font_manager').disabled = True
    max_arithmetic_intensity = 6
    max_computing_power = 0

    # Scalability
    platforms_config['UPMEMx16'], platforms_config['UPMEMx24'], platforms_config['UPMEMx32'], platforms_config['GPUx2'] = {}, {}, {}, {}
    platforms_config['UPMEMx16']['computing_power'], platforms_config['UPMEMx16']['bandwidth'] = platforms_config['UPMEM']['computing_power'] * 16, platforms_config['UPMEM']['bandwidth'] * 16
    platforms_config['UPMEMx24']['computing_power'], platforms_config['UPMEMx24']['bandwidth'] = platforms_config['UPMEM']['computing_power'] * 24, platforms_config['UPMEM']['bandwidth'] * 24
    platforms_config['UPMEMx32']['computing_power'], platforms_config['UPMEMx32']['bandwidth'] = platforms_config['UPMEM']['computing_power'] * 32, platforms_config['UPMEM']['bandwidth'] * 32
    platforms_config['GPUx2']['computing_power'], platforms_config['GPUx2']['bandwidth'] = platforms_config['GPU']['computing_power'] * 2, platforms_config['GPU']['bandwidth'] * 2
    platforms_config.pop('UPMEM')

    _, ax = plt.subplots()
    for name, config in platforms_config.items():
        roof_x = config['computing_power'] / config['bandwidth']
        if roof_x < max_arithmetic_intensity:
            xs = [0, config['computing_power'] / config['bandwidth'], max_arithmetic_intensity]
            ys = [0, config['computing_power'], config['computing_power']]
        else:
            xs = [0, max_arithmetic_intensity]
            ys = [0, config['bandwidth'] * max_arithmetic_intensity]
        if ys[-1] > max_computing_power:
            max_computing_power = ys[-1]
        ax.plot(xs, ys, '-', label = name)
    for name, intensity in arithmetic_intensity.items():
        xs = [intensity] * 2
        ys = [0, (max_computing_power // 1000 + 1) * 1000]
        ax.plot(xs, ys, '--', label = name)

    ax.set_xlabel('Arithmetic Intensity (OPs/Byte)')
    ax.set_ylabel('GOPs / s')
    # ax.set_title(f'Computing Power - Arithmetic Intensity')
    plt.legend()
    plt.savefig(plot_file_name)
    getLogger('matplotlib.font_manager').disabled = getLogger_disabled  # Recover logger.


if __name__ == '__main__':
    # Get configure parameters.
    parser = argparse.ArgumentParser()
    # Disk file params
    parser.add_argument('--datasets-config-file-name', type=str, default='datasets_config.json', help='The full name, i.e. path + name, of the input json file of dataset configurations (default: datasets_config.json)')
    parser.add_argument('--platforms-config-file-name', type=str, default='platforms_config.json', help='The full name, i.e. path + name, of the input json file of platform configurations (default: platforms_config.json)')
    parser.add_argument('--result-excel-file-name', type=str, default='result.xlsx', help='The full name, i.e. path + name, of the output result excel workbook based on the data in the input log file (default: result.xlsx)')
    parser.add_argument('--result-plot-file-name', type=str, default='result.png', help='The full name, i.e. path + name, of the output result diagram based on the data in the input log file (default: result.png)')
    args = parser.parse_args()

    # Record configure params.
    for arg_key in parser._actions[1:]:
        arg_key = arg_key.dest
        print(('{}: {}').format(arg_key, eval('args.{}'.format(arg_key))))

    # Load configures.
    datasets_config, platforms_config = {}, {}
    with open(args.datasets_config_file_name, 'r') as datasets_config_file:
        datasets_config = json.load(datasets_config_file)
    with open(args.platforms_config_file_name, 'r') as platforms_config_file:
        platforms_config = json.load(platforms_config_file)
    arithmetic_intensity = roofline_model_datasets(**datasets_config)

    draw_plots(platforms_config, arithmetic_intensity, args.result_plot_file_name)
    
    # Record the arithmetic intensity of datasets as well as configurations of platforms.
    result_excel = Workbook()
    sheet = result_excel.add_sheet('Arithmetic Intensity')
    sheet.write(0, 0, label = 'Dataset')
    sheet.write(0, 1, label = 'Arithmetic Intensity')
    for row, (name, intensity) in enumerate(arithmetic_intensity.items(), 1):
        sheet.write(row, 0, label = name)
        sheet.write(row, 1, label = intensity)
    sheet = result_excel.add_sheet('Platform Configurations')
    sheet.write(0, 0, label = 'Platform')
    sheet.write(0, 1, label = 'Computing Power')
    sheet.write(0, 2, label = 'Bandwidth')
    for row, (name, config) in enumerate(platforms_config.items(), 1):
        sheet.write(row, 0, label = name)
        sheet.write(row, 1, label = config['computing_power'])
        sheet.write(row, 2, label = config['bandwidth'])
    result_excel.save(args.result_excel_file_name)
