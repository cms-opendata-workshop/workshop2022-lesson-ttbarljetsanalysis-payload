import uproot
import hist
import mplhep as hep
import matplotlib.pyplot as plt

file = uproot.open("histograms.root")
file.classnames()
data = file['4j2b_data'].to_hist()
ttbar = file['4j2b_ttbar'].to_hist()
single_atop = file['4j2b_single_atop_t_chan'].to_hist()
single_top = file['4j2b_single_top_t_chan'].to_hist()
single_tW = file['4j2b_single_top_tW'].to_hist()
wjets = file['4j2b_wjets'].to_hist()
bklabels = ["ttbar","sigle_atop","single_top","single_tW","wjets"]

hep.style.use("CMS")
hep.cms.label("open data",data=True, lumi=2.26, year=2015)
hep.histplot(data,histtype="errorbar", color='k', capsize=4, label="Data")
hep.histplot([ttbar,single_atop,single_top,single_tW,wjets],stack=True, histtype='fill', label=bklabels, sort='yield')
plt.legend(frameon=False)
plt.xlabel("$m_{bjj}$ [Gev]");
plt.savefig('finalplot.png')
plt.show()