# TTBarLJets analysis using Coffea

This example is an adaptation of the [example analysis](https://github.com/iris-hep/analysis-grand-challenge/blob/main/analyses/cms-open-data-ttbar/coffea.ipynb) presented at the [IRIS-HEP AGC Workwhop 2022](https://indico.cern.ch/e/agc-tools-2)

## Additional packages

In order to run the simplified analysis with coffea, we use the [python tools docker container](https://cms-opendata-workshop.github.io/workshop2022-lesson-docker/03-docker-for-cms-opendata/index.html#python-tools-container), which was prepared for the CMS Open Data Workshop 2022.

This container alread had `uproot`, `awkward`, `numpy` and `xrootd` installed.  We need a few extra tools that we will be needing.  One can install them, directly into the container with:


`pip install vector hist mplhep coffea`


## First dummy test

To run a simple procesing check, simply do:

`python coffeaAnalysis_ttbarljets.py`

This will run a currently-failing version of an analysis that will be modified for the workshop.  The processing part, however, run somoothly, as so it is good for testing remote or local acces of files (whose links are stored in the `ntuples.json` file).
