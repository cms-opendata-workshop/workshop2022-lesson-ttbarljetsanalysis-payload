import uproot
import numpy as np
import awkward as ak
import hist
import matplotlib.pyplot as plt
import mplhep as hep
from coffea.nanoevents import NanoEventsFactory, BaseSchema
from agc_schema import AGCSchema

#events = NanoEventsFactory.from_root('root://eospublic.cern.ch//eos/opendata/cms/upload/POET/23-Jul-22/RunIIFall15MiniAODv2_TT_TuneCUETP8M1_13TeV-powheg-pythia8_flat/6D5A8366-FEA3-4C76-BD4E-22B56D367281_flat.root', schemaclass=AGCSchema, treepath='events').events()
#events = NanoEventsFactory.from_root('root://eospublic.cern.ch//eos/opendata/cms/upload/POET/23-Jul-22/Run2015D_SingleElectron_flat.root', schemaclass=AGCSchema, treepath='events').events()
#events = NanoEventsFactory.from_root('root://eospublic.cern.ch//eos/opendata/cms/upload/POET/23-Jul-22/Run2015D_SingleMuon_flat.root', schemaclass=AGCSchema, treepath='events').events()
events = NanoEventsFactory.from_root('root://eospublic.cern.ch//eos/opendata/cms/upload/POET/23-Jul-22/RunIIFall15MiniAODv2_TT_TuneCUETP8M1_13TeV-powheg-pythia8_flat/00EFE6B3-01FE-4FBF-ADCB-2921D5903E44_flat.root', schemaclass=AGCSchema, treepath='events').events()


selected_electrons = events.electron[(events.electron.pt > 30) & (abs(events.electron.eta)<2.1) & (events.electron.isTight == True) & (events.electron.sip3d < 4)]
selected_muons = events.muon[(events.muon.pt > 30) & (abs(events.muon.eta)<2.1) & (events.muon.isTight == True) & (events.muon.sip3d < 4) & (events.muon.pfreliso04DBCorr < 0.15)]
selected_jets = events.jet[(events.jet.corrpt > 30) & (abs(events.jet.eta)<2.4)]

#event filters
event_filters = ((ak.count(selected_electrons.pt, axis=1) + ak.count(selected_muons.pt, axis=1)) == 1)
event_filters = event_filters & (ak.count(selected_jets.corrpt, axis=1) >= 4)
B_TAG_THRESHOLD = 0.8
event_filters = event_filters & (ak.sum(selected_jets.btag >= B_TAG_THRESHOLD, axis=1) >= 1)
selected_events = events[event_filters]
selected_electrons = selected_electrons[event_filters]
selected_muons = selected_muons[event_filters]
selected_jets = selected_jets[event_filters]

#4j2b region filter
region_filter = ak.sum(selected_jets.btag > B_TAG_THRESHOLD, axis=1) >= 2
selected_jets_region = selected_jets[region_filter]

# reconstruct hadronic top as bjj system with largest pT
trijet = ak.combinations(selected_jets_region, 3, fields=["j1", "j2", "j3"])  # trijet candidates
trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3  # calculate four-momentum of tri-jet system
trijet["max_btag"] = np.maximum(trijet.j1.btag, np.maximum(trijet.j2.btag, trijet.j3.btag))
trijet = trijet[trijet.max_btag > B_TAG_THRESHOLD]  # require at least one-btag in trijet candidates
# pick trijet candidate with largest pT and calculate mass of system
trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass
observable = ak.flatten(trijet_mass)

#fill in a histogram
histogram = hist.Hist.new.Reg(25, 50, 550, name="observable", label="observable [GeV]").StrCat(["4j1b", "4j2b"], name="region", label="Region").StrCat([], name="process", label="Process", growth=True).StrCat([], name="variation", label="Systematic variation", growth=True).Weight()
histogram.ndim
histogram.axes
histogram[:,1,:,:]
histogram.fill(observable=observable, region="4j2b", process="ttbar", variation="nominal", weight=1)
#plot
histogram[:,"4j2b","ttbar","nominal"].plot(histtype="fill", linewidth=1, edgecolor="grey", label='ttbar')
plt.legend()
plt.title(">= 4 jets, >= 2 b-tags")
plt.xlabel("$m_{bjj}$ [Gev]")
plt.show()