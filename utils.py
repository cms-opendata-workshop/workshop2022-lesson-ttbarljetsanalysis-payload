# https://pypi.org/project/hist/
import hist

# https://docs.python.org/3/library/json.html
import json

# https://matplotlib.org/stable/index.html
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep

#https://numpy.org/doc/stable/
import numpy as np

#https://uproot.readthedocs.io/en/latest/basic.html
import uproot

#----------------------
def set_style():
#---------------------
    mpl.style.use("ggplot")
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["axes.edgecolor"] = "222222"
    plt.rcParams["axes.labelcolor"] = "222222"
    plt.rcParams["xtick.color"] = "222222"
    plt.rcParams["ytick.color"] = "222222"
    plt.rcParams["font.size"] = 12
    plt.rcParams['text.color'] = "222222"

#----------------------------------------------------------------
def construct_fileset(n_files_max_per_sample, use_xcache=False):
#----------------------------------------------------------------
    # using cross sections from CMS
    # x-secs are in pb
    xsec_info = {
        "ttbar": 831.76,
        "single_atop_t_chan": 26.38,
        "single_top_t_chan": 44.33,
        "single_top_tW": 35.6 + 36.6,
        "wjets": 61526.7,
        "data": None
    }

    # get the avg scaling factor for datasets because we haven't
    # recovered the original number of events that went into each file
    # the skimming jobs were not 1 to 1
    with open("ntuples_factors.json") as nf:
        nfactors_info = json.load(nf)

    nfactors_dic = {}
    for process in nfactors_info.keys():
        nfactors_dic[process] = {}
        for variation in nfactors_info[process].keys():
            nfactors_dic[process][variation]= nfactors_info[process][variation]["factor"]

    # list of files
    with open("ntuples.json") as f:
        file_info = json.load(f)

    # process into "fileset" summarizing all info
    fileset = {}
    for process in file_info.keys():
        #if process == "data":
        #    continue  # skip data
        for variation in file_info[process].keys():
            file_list = file_info[process][variation]["files"]
            if n_files_max_per_sample != -1:
                file_list = file_list[:n_files_max_per_sample]  # use partial set of samples

            file_paths = [f["path"] for f in file_list]
            #nevts_total = sum([f["nevts"] for f in file_list])
            nevts_total = sum([f["nevts"] for f in file_list])/nfactors_dic[process][variation]    
            metadata = {"process": process, "variation": variation, "nevts": nevts_total, "xsec": xsec_info[process]}
            fileset.update({f"{process}__{variation}": {"files": file_paths, "metadata": metadata}})

    return fileset

#-------------------------------------------------------
def save_histograms(all_histograms, fileset, filename):
#-------------------------------------------------------
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    all_histograms += 1e-6  # add minimal event count to all bins to avoid crashes when processing a small number of samples

    pseudo_data = (all_histograms[:, :, "ttbar", "ME_var"] + all_histograms[:, :, "ttbar", "PS_var"]) / 2  + all_histograms[:, :, "wjets", "nominal"]

    with uproot.recreate(filename) as f:
        for region in ["4j1b", "4j2b"]:
            f[f"{region}_pseudodata"] = pseudo_data[120j::hist.rebin(2), region]
            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                f[f"{region}_{sample_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, "nominal"]

                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, variation_name]

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, variation_name]

            f[f"{region}_ttbar_ME_var"] = all_histograms[120j::hist.rebin(2), region, "ttbar", "ME_var"]
            f[f"{region}_ttbar_PS_var"] = all_histograms[120j::hist.rebin(2), region, "ttbar", "PS_var"]
            for process in ["ttbar", "wjets"]:
                f[f"{region}_{process}_scaledown"] = all_histograms[120j::hist.rebin(2), region, process, "scaledown"]
                f[f"{region}_{process}_scaleup"] = all_histograms[120j::hist.rebin(2), region, process, "scaleup"]

