#!/bin/bash
# $0 <input> <typedef-out> <ompd.h-out>
# keep block comments, and api function typedefs
sed '/^\/\*/{:b1;N;/\*\//!bb1;p};/^typedef.*_apifn_t/{:b2;N;/);/!bb2;s/_apifn_t/_fn_t/;p};d' $1 > $2
# remove api function typedefs
sed '/^typedef.*_apifn_t/{:b2;N;/);/!bb2;d}' $1 > $3
