#!/bin/csh

if ($#argv != 6) then
        echo "Usage: $0 site mm dd yyyy btime etime"
	echo "   where"
	echo "     ssss = site (ex. KMLB)"
	echo "     mm = month (ex. 01)"
	echo "     dd = day (ex. 12)"
	echo "     yyyy = year (ex. 2012)"
        echo "     tttt = btime (ex. 0000)"
        echo "     tttt = etime (ex. 2359)"
	exit
endif

 set site = $1
 set month = $2
 set day = $3
 set year = $4
 set btime = $5
 set etime = $6

cd /gvs2/gpmgv_data/gpmgv/$site/raw/$year/$month$day/
 
foreach f (`ls *`)
  echo $f

  set ftime = `echo $f | cut -d'_' -f2-4`
  set time = `echo $ftime | cut -c1-4`
# echo "ftime: $ftime"
  set itime = `echo $time | bc` 
# echo "itime: $itime"
  
  if ($itime < 100) then
      set ittext = "00"$itime
  endif
  if ($itime >= 100 && $itime < 1000) then
      set ittext = "0"$itime
  endif
  if ($itime >= 1000) then
      set ittext = $itime
  endif
  if ($itime >= 0 && $itime < 10) then
        set ittext = "000"$itime
  endif
  
  echo "ftime: $ftime"
  echo "itime: $itime"
  echo "btime: $btime"
  echo "etime: $etime"

  if ($itime >= $btime && $itime <= $etime) then
 
  cd /gpmgv1/NPOL/PyArt/DPQCV0.2/
  python DPQC.py --file /gvs2/gpmgv_data/gpmgv/KMLB/raw/2020/0606/KMLB20200606_120129_V06 --thresh_dict KMLB_200606_VN_dict-1.txt

  endif

end

echo " "
echo "xv /gvs2/gpmgv_data/gpmgv/$site/images/$year/$month$day/$site* &"
