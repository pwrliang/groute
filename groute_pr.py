#!/usr/bin/env python3
import os
import sys

work_path = os.path.abspath(os.path.join(sys.argv[0], os.pardir))
pr_bin_path = os.path.join(work_path, 'cmake-build-debug/pr')

if not os.path.exists(pr_bin_path):
    print("binary[%s] doesn't exists" % pr_bin_path, file=sys.stderr)
    exit(1)

url_list = ["http://164.107.116.109/dataset/kron_g500-logn21_undirected_weighted.gr",
            "http://164.107.116.109/dataset/road_usa_undirected_weighted.gr"]


for url in url_list:
    data_filename = url.split('/')[-1]
    data_path = os.path.join(work_path, data_filename)
    log_path = os.path.join(work_path, 'pr_%s.groute' % (data_filename))

    if not os.path.exists(data_path):
        os.system('wget -P %s %s' % (work_path, url))

    os.system('%s -single -graphfile %s -wl_sort -wl_alloc_factor=1 > %s' % (pr_bin_path, data_path, log_path))


