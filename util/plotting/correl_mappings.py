# This file is eval'd inside the plot-correlation.py file

# This maps the named GPGPU-Sim config to the card name reported in the nvprof file.
#   Every time you want to correlate a new configuration, you need to map it here.
config_maps = \
{
    "PUB_TITANX": "TITAN X (Pascal)",
    "TITANX_P102": "TITAN X (Pascal)",
    "TITANX": "TITAN X (Pascal)",
    "TITANV": "TITAN V",
    "3.x_PASCALTITANX" : "TITAN X (Pascal)",
    "3.x_P100" :  "Tesla P100",
    "P100_HBM" : "Tesla P100",
    "GTX480" : "GeForce GTX 480",
    "GTX1080Ti" : "GeForce GTX 1080 Ti",
}


# Every stat you want to correlate gets an entry here.
#   For cycles, the math is different for every card so we have differnt stats baed on the hardware.
import collections
CorrelStat = collections.namedtuple('CorrelStat', 'chart_name hw_eval hw_error sim_eval hw_name plotfile')
correl_list = \
[
    # 1417 MHz
    CorrelStat(chart_name="Cycles",
        plotfile="titanv-cycles.html",
        hw_eval="np.average(hw[\"Duration\"])*1200",
        hw_error="np.max(hw[\"Duration\"])*1200 - np.average(hw[\"Duration\"])*1200,"+\
                 "np.average(hw[\"Duration\"])*1200 - np.min(hw[\"Duration\"])*1200",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="TITAN V"
    ),
    # 1417 MHz
    CorrelStat(chart_name="Cycles",
        plotfile="titanx-p102-cycles.html",
        hw_eval="np.average(hw[\"Duration\"])*1417",
        hw_error="np.max(hw[\"Duration\"])*1417 - np.average(hw[\"Duration\"])*1417,"+\
                 "np.average(hw[\"Duration\"])*1417 - np.min(hw[\"Duration\"])*1417",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="TITAN X (Pascal)"
    ),
    # (1400 MHz - 16-wide SIMD)
    CorrelStat(chart_name="Cycles",
        plotfile="gtx480-cycles.html",
        hw_eval="np.average(hw[\"Duration\"])*1400",
        hw_error="np.max(hw[\"Duration\"])*1400 - np.average(hw[\"Duration\"])*1400,"+\
                 "np.average(hw[\"Duration\"])*1400 - np.min(hw[\"Duration\"])*1400",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])*2",
        hw_name="GeForce GTX 480"
    ),
    # 1480 MHz
    CorrelStat(chart_name="Cycles",
        plotfile="p100-cycles.html",
        hw_eval="np.average(hw[\"Duration\"])*1480",
        hw_error="np.max(hw[\"Duration\"])*1480 - np.average(hw[\"Duration\"])*1480,"+\
                 "np.average(hw[\"Duration\"])*1480 - np.min(hw[\"Duration\"])*1480",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="Tesla P100"
    ),
    # 1480 MHz
    CorrelStat(chart_name="Cycles",
        plotfile="1080ti-cycles.html",
        hw_eval="np.average(hw[\"Duration\"])*1480",
        hw_error="np.max(hw[\"Duration\"])*1480 - np.average(hw[\"Duration\"])*1480,"+\
                 "np.average(hw[\"Duration\"])*1480 - np.min(hw[\"Duration\"])*1480",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="GeForce GTX 1080 Ti"
    ),

    # Common, non-cycle stats
    CorrelStat(chart_name="Warp Instructions",
        plotfile="warp-inst.html",
        hw_eval="np.average(hw[\"inst_executed\"])",
        hw_error=None,
        sim_eval="float(sim[\"gpgpu_n_tot_w_icount\s*=\s*(.*)\"])",
        hw_name="all"
    ),
    CorrelStat(chart_name="L2 read hits",
        plotfile="l2-read-hits.html",
        hw_eval="np.average(hw[\"l2_tex_read_transactions\"])*np.average(hw[\"l2_tex_read_hit_rate\"])/100",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])",
        hw_name="all"
    ),
    CorrelStat(chart_name="L2 read transactions",
        plotfile="l2-read-transactions.html",
        hw_eval="np.average(hw[\"l2_tex_read_transactions\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all"
    ),
    CorrelStat(chart_name="DRAM read transactions",
        plotfile="dram-read-transactions.html",
        hw_eval="np.average(hw[\"dram_read_transactions\"])",
        hw_error=None,
        sim_eval="float(sim[\"Read\s*=\s*(.*)\"])+float(sim[\"L2_Alloc\s*=\s*(.*)\"])",
        hw_name="all"
    ),
    CorrelStat(chart_name="L2 write transactions",
        plotfile="l2-write-transactions.html",
        hw_eval="np.average(hw[\"l2_tex_write_transactions\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all"
    ),
    CorrelStat(chart_name="DRAM Reads",
        plotfile="dram-read-transactions.html",
        hw_eval="np.average(hw[\"dram_read_transactions\"])",
        hw_error=None,
        sim_eval="float(sim[\".*n_rd\s*=\s*([0-9]+).*\"])*12",
        hw_name="all"
    ),
    CorrelStat(chart_name="L2 BW",
        plotfile="l2_bw.html",
        hw_eval="np.average(hw[\"l2_tex_read_throughput\"])",
        hw_error=None,
        sim_eval="float(sim[\"L2_BW\s*=\s*(.*)GB/Sec\"])",
        hw_name="all"
    ),
    CorrelStat(chart_name="L2 read Hit rate",
        plotfile="l2-read-hitrate.html",
        hw_eval="np.average(hw[\"l2_tex_read_hit_rate\"])",
        hw_error=None,
        sim_eval=
            "100*float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])/"+\
            "float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all"
    ),
    CorrelStat(chart_name="L2 write Hit rate",
        plotfile="l2-write-hitrate.html",
        hw_eval="np.average(hw[\"l2_tex_write_hit_rate\"])",
        hw_error=None,
        sim_eval=
            "100*float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\]\s*=\s*(.*)\"])/"+\
            "float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all"
    ),
    CorrelStat(chart_name="Occupancy",
        plotfile="occupancy.html",
        hw_eval="np.average(hw[\"achieved_occupancy\"])*100",
        hw_error=None,
        sim_eval="float(sim[\"gpu_occupancy\s*=\s*(.*)%\"])",
        hw_name="all"
    ),
]
