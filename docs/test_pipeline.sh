#!/bin/bash

mkdir -p parascopy/data
cd parascopy/data

echo -e "\n\nDownloading homology tables\n\n"
wget -c https://dl.dropboxusercontent.com/s/okzeedb6gze6zzs/homology_table_hg38.tar
tar xf homology_table_hg38.tar

echo -e "\n\nDownloading precomputed model parameters\n\n"
wget -c https://dl.dropboxusercontent.com/s/5fsohggje778dlb/models_v1.2.5.tar.gz
tar xzf models_v1.2.5.tar.gz

echo -e "\n\nDownloading reference genome\n\n"
wget -c -O hg38.fa \
    ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa
wget -c -O hg38.fa.fai \
    ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai

cd ..
echo -e "\n\nDownloading single-sample data (360 Mb)\n\n"
wget -c https://dl.dropboxusercontent.com/s/46tt30brotjml0y/HG00113.cram
samtools index HG00113.cram

# Takes 2-5 minutes.
echo -e "\n\nCalculating background read depth\n\n"
parascopy depth -i HG00113.cram -f data/hg38.fa -g hg38 -o depth_HG00113

# Takes 10-40 minutes depending on the number of threads.
echo -e "\n\nCalculating copy number profiles for a single sample\n\n"
parascopy cn-using data/models_v1.2.5/EUR \
    -i HG00113.cram -f data/hg38.fa -t data/homology_table/hg38.bed.gz \
    -d depth_HG00113 -o parascopy_HG00113
# Optionally: specify number of threads with -@ command.
# You can find output file descriptions here: https://github.com/tprodanov/parascopy/blob/main/docs/cn_output.md

echo =e "\n\nDownloading multi-sample data (195 Mb)\n\n"
wget -c https://dl.dropboxusercontent.com/s/o4ntonnxhs780ui/GBA.tar.gz
tar xvf GBA.tar.gz

# Takes 25-30 minutes.
echo -e "\n\nCalculating copy number profiles for multiple samples\n\n"
parascopy cn -I GBA/input.list  -f data/hg38.fa \
    -t data/homology_table/hg38.bed.gz -d GBA/1kgp_depth.csv.gz \
    -r chr1:155231479-155244699::GBA -o parascopy_GBA
# -r : region in format "chr:start-end[::name]".
# Optionally, you can supply regions in a BED file with -R regions.bed
