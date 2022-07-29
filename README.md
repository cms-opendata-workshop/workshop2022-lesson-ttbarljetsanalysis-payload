# TTBarLJets analysis using Coffea

This example is an adaptation of the [example analysis](https://github.com/iris-hep/analysis-grand-challenge/blob/main/analyses/cms-open-data-ttbar/coffea.ipynb) presented at the [IRIS-HEP AGC Workwhop 2022](https://indico.cern.ch/e/agc-tools-2)

## Additional packages

In order to run the simplified analysis with coffea, we use the [python tools docker container](https://cms-opendata-workshop.github.io/workshop2022-lesson-docker/03-docker-for-cms-opendata/index.html#python-tools-container), which was prepared for the CMS Open Data Workshop 2022.

This container alread had `uproot`, `awkward`, `numpy` and `xrootd` installed.  We need a few extra tools.  One can install them, directly into the container with:


`pip install vector hist mplhep coffea`


## Working examples

* To run a very simple version of the essence of this analysis, without systematics, and over one simple dataseta and analysis region in order to produce a single observable plot:

    `python coffeaAnalysis_basics.py`

   Ordered pieces of it, from top to bottom, can be better run on a jupyter notebook.  Or, if preferred, run interactively:

   `python -i coffeaAnalysis_basics.py`


* To run the full analysis but over a limited `ntuples.json` file (with just one input file per dataset), do:

    `python coffeaAnalysis_ttbarljets.py`

    Final images for histograms are not saved well, but the final `histograms.root` should be sound.  This is the file which will be used later with cabinetry.
