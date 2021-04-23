#!/bin/csh

setenv TSDISTK /home/trmmgv/trmm/toolkit
setenv LD_LIBRARY_PATH /home/trmmgv/trmm/lib
setenv GVS_DATA_PATH /home/trmmgv/trmm/data
setenv GVS_DB_PATH /home/trmmgv/trmm/data
setenv IDL_PATH '<IDL_DEFAULT>:/home/trmmgv/idl_tools/rsl_in_idl:/home/trmmgv/realtime_bin'
setenv IDL_STARTUP /home/trmmgv/idl_tools/start.pro

set file  = $1

if ($#argv != 1) then
    echo ""
    echo "Usage: "
    echo "     convert_to_cf file "
    echo ""
    echo "     file = full path of file"
    echo ""
    exit
endif

source /usr/local/exelis/idl/bin/idl_setup

idl<<EOF
  convert_to_cf, '$file'
EOF

