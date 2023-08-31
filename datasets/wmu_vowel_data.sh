#!/bin/bash
#
# This script download the WMU dataset from its website.
# From https://homepages.wmich.edu/~hillenbr/voweldata.html
#
# Data are merged and uncompressed
#
# It takes the folder name as argument
#


# Remove last slash from argument
FOLDER_NAME=${1%/}


if ! mkdir -p $FOLDER_NAME
then
  echo "Can't create directory '$FOLDER_NAME'"
  exit
fi

cd $FOLDER_NAME

curl -OL https://homepages.wmich.edu/~hillenbr/voweldata/men.zip
curl -OL https://homepages.wmich.edu/~hillenbr/voweldata/women.zip
curl -OL https://homepages.wmich.edu/~hillenbr/voweldata/kids.zip

mkdir -p men women kids
unzip men.zip -d men/
unzip women.zip -d women/
unzip kids.zip -d kids/

rm men.zip women.zip kids.zip
