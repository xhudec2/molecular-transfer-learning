#!/bin/sh

mkdir -p data

gdown "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv" -O data/qm7.csv
gdown "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv" -O data/Lipophilicity.csv

# takes around 5-10 minutes
git clone https://github.com/davidbuterez/mf-pcba.git
uv run mf-pcba/pubchem_retrieve.py --AID "504329" --list_of_sd_cols "Primary Screen Percent Inhibition @ 12.5 uM Rep 1" "Primary Screen Percent Inhibition @ 12.5 uM Rep 2" "Primary Screen Percent Inhibition @ 12.5 uM Rep 3" --list_of_dr_cols "IC50" --transform_dr "pXC50" --save_dir data

mv data/AID504329/SD.csv data/SD.csv
# cleanup
rm -r data/AID504329
rm -rf mf-pcba
