#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Produce the ntuples.json file needed for the
analysis. 

NOTE: This script needs to be run in an environment where
EOS is installed (lxplus or SWAN will work)

'''

import json
import os
import uproot
import yaml

from subprocess import (
    PIPE,
    run
)

eos_env = 'root://eospublic.cern.ch'
os.environ["EOS_MGM_URL"] = eos_env

poet_path = '/eos/opendata/cms/upload/POET/23-Jul-22/'

contents = run(
    f"xrdfs {eos_env} ls {poet_path}",
    stdout=PIPE,
    stderr=PIPE,
    universal_newlines=True,
    shell=True
)

contents = contents.stdout.split('\n')[:-1]


# In[ ]:


# These are the dirs which contain the files we want
data_dirs = [d for d in contents if '_flat' in d and '.root' not in d]

# These dirs contain the metadata which we can query as a check (re: number of events)
# We will query the root files directly later to determine number of events
metadata_dirs = [d for d in contents if not '_flat' in d and '.root' not in d]


# In[ ]:


# See what's in the metadata yaml files
metadata = run(
    f"xrdfs {eos_env} cat {poet_path}/RunIIFall15MiniAODv2_WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/metadata.yaml",
    stdout=PIPE,
    stderr=PIPE,
    universal_newlines=True,
    shell=True
)

md = yaml.safe_load(
    metadata.stdout
)


# In[ ]:


# How to get the number of events in the miniAOD?

with uproot.open(
                'root://eospublic.cern.ch//eos/opendata/cms/mc/RunIIFall15MiniAODv2/WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/PU25nsData2015v1_76X_mcRun2_asymptotic_v12_ext4-v1/40000/2C8FC19B-01EA-E511-BE10-BC305B390AB4.root'
) as f:
    print(
        f.keys()
    )
    print(
        f'{f["Events"].num_entries} events'
    )


# In[ ]:


ntuples = {
    "data": {},
    "ttbar": {},
    "single_atop_t_chan": {},
    "single_top_t_chan": {},
    "single_top_tW": {},
    "wjets": {}
}


# In[ ]:


def get_files(process_dir):
    
    file_names = []
    
    for d in process_dir:
    
        files = run(
                    f"xrdfs {eos_env} ls {poet_path}{d}",
                    stdout=PIPE,
                    stderr=PIPE,
                    universal_newlines=True,
                    shell=True
                )
    
        fns = files.stdout.split('\n')[:-1]
        file_names += [f"{eos_env}/{fn}" for fn in fns]
    
    return file_names


def num_events_list(files):
    
    num_events = []
    
    for i, file_name in enumerate(files):
        
        with uproot.open(file_name) as f:
            
            num_events.append(
                f["events"].num_entries
            )
            
    return num_events


def update_dict(process, variation, process_dir):
    
        print(
            process_dir
        )
    
        files = get_files(process_dir)
        nevts_list = num_events_list(files)
    
        ntuples[process].update(
            {
                variation: {
                "files": [{"path": f, "nevts": n} for f, n in zip(files, nevts_list)],
                "nevts_total": sum(nevts_list),
            }
        }
        
    )


# In[ ]:


update_dict(
    "data", "nominal", 
    [
        "Run2015D_SingleMuon_flat", 
        "Run2015D_SingleElectron_flat"
    ]
)

update_dict("ttbar", "nominal", ["RunIIFall15MiniAODv2_TT_TuneCUETP8M1_13TeV-powheg-pythia8_flat"])

update_dict("ttbar", "scaledown", ["RunIIFall15MiniAODv2_TT_TuneCUETP8M1_13TeV-powheg-scaledown-pythia8_flat"])

update_dict("ttbar", "scaleup", ["RunIIFall15MiniAODv2_TT_TuneCUETP8M1_13TeV-powheg-scaleup-pythia8_flat"])

update_dict("ttbar", "PS_var", ["RunIIFall15MiniAODv2_TT_TuneEE5C_13TeV-powheg-herwigpp_flat"])

update_dict("ttbar", "ME_var", ["RunIIFall15MiniAODv2_TTJets_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_flat"])

update_dict("single_top_t_chan", "nominal", ["RunIIFall15MiniAODv2_ST_t-channel_4f_leptonDecays_13TeV-amcatnlo-pythia8_TuneCUETP8M1_flat"])

update_dict("single_atop_t_chan", "nominal", ["RunIIFall15MiniAODv2_ST_t-channel_antitop_4f_leptonDecays_13TeV-powheg-pythia8_TuneCUETP8M1_flat"])

update_dict(
    "single_top_tW", "nominal", 
    [
        "RunIIFall15MiniAODv2_ST_tW_top_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1_flat", 
        "RunIIFall15MiniAODv2_ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1_flat"
    ]
)

update_dict("wjets", "nominal", ["RunIIFall15MiniAODv2_WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_flat"])


# In[ ]:


output_file_name = 'ntuples.json'

json.dump(
    ntuples,
    open(output_file_name, 'w'),
    sort_keys=False,
    indent=4
)

print(
    f"Output written to {output_file_name}"
)


# In[ ]:




