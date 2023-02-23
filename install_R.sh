#!/bin/bash
apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
apt-get -qq install dialog apt-utils -y
apt-get install apt-transport-https -y
apt-get install -qq software-properties-common -y
apt-get -qq update
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' -y
apt-get -qq update

apt-get -qq install r-base -y
apt-get -qq install libssl-dev -y
apt-get -qq install libgmp3-dev  -y
apt-get -qq install git -y
apt-get -qq install build-essential  -y
apt-get -qq install libv8-dev  -y
apt-get -qq install libcurl4-openssl-dev -y
apt-get -qq install libgsl-dev -y
apt-get -qq install libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev -y
apt-get -qq install libxml2 libxml2-dev libfontconfig1-dev libcurl4-gnutls-dev libfribidi-dev libharfbuzz-dev --yes

Rscript -e 'install.packages(c("V8"),repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
Rscript -e 'install.packages(c("sfsmisc"),repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
Rscript -e 'install.packages(c("clue"),repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/randomForest/randomForest_4.6-14.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages(c("lattice"),repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
Rscript -e 'install.packages(c("devtools"),repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
Rscript -e 'install.packages(c("MASS"),repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
Rscript -e 'install.packages("BiocManager")'
Rscript -e 'BiocManager::install(c("igraph"))'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/fastICA/fastICA_1.2-2.tar.gz", repos=NULL, type="source")'
Rscript -e 'BiocManager::install(c("SID", "bnlearn", "pcalg", "kpcalg", "glmnet", "mboost"))'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/CAM/CAM_1.0.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/sparsebnUtils_0.0.8.tar.gz", repos=NULL, type="source")'
Rscript -e 'BiocManager::install(c("ccdrAlgorithm", "discretecdAlgorithm"))'

apt-get -qq install libxml2-dev -y
Rscript -e 'install.packages("devtools")'
Rscript -e 'library(devtools); install_github("cran/CAM"); install_github("cran/momentchi2"); install_github("Diviyan-Kalainathan/RCIT", quiet=TRUE, verbose=FALSE)'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/sparsebn/sparsebn_0.1.2.tar.gz", repos=NULL, type="source")'