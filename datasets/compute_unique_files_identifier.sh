#!/bin/sh
# Compute a unique identifier given a collection of files.
#
# This identifier is function of the content of the files.
# It do not depend of the files name, and do not depend of the
# order of the arguments.
#
# To compute this identifier:
# We take the sha1 checksum of all files and sort them by alphabetical order.
# We them build a string from this checksum separated by a "-".
# Finaly, we hash this string again with sha1.


if [ $# -eq 0 ]
then
  echo "Usage: ./"$0" file1 file2 ..."
  exit 1
fi

shasum -a 1 $@ | cut -d ' ' -f 1 | sort | paste -s -d '-'  - | shasum -a 1 | cut -d ' ' -f 1
