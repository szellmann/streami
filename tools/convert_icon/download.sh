vars='U V W'
#for var in $vars;
#do
#  #for i in $(seq 1 120);
#  for i in $(seq 1 5);
#  do
#    wget https://opendata.dwd.de/weather/nwp/icon/grib/00/${var,,}/icon_global_icosahedral_model-level_2026020200_000_${i}_${var}.grib2.bz2
#    bzip2 -d icon_global_icosahedral_model-level_2026020200_000_${i}_${var}.grib2.bz2
#    rm -rf icon_global_icosahedral_model-level_2026020200_000_${i}_${var}.grib2.bz2
#    cdo -f nc copy icon_global_icosahedral_model-level_2026020200_000_${i}_${var}.grib2 icon_global_icosahedral_model-level_2026020200_000_${i}_${var}.nc
#    rm -rf icon_global_icosahedral_model-level_2026020200_000_${i}_${var}.grib2
#  done
#done

#for i in $(seq 1 120);
for i in $(seq 1 5);
do
  wget https://opendata.dwd.de/weather/nwp/icon/grib/00/hhl/icon_global_icosahedral_time-invariant_2026020200_${i}_HHL.grib2.bz2
  bzip2 -d icon_global_icosahedral_time-invariant_2026020200_${i}_HHL.grib2.bz2
  rm -rf icon_global_icosahedral_time-invariant_2026020200_${i}_HHL.grib2.bz2
  cdo -f nc copy icon_global_icosahedral_time-invariant_2026020200_${i}_HHL.grib2 icon_global_icosahedral_time-invariant_2026020200_${i}_HHL.nc
  rm -rf icon_global_icosahedral_time-invariant_2026020200_${i}_HHL.grib2
done

wget https://opendata.dwd.de/weather/nwp/icon/grib/00/hsurf/icon_global_icosahedral_time-invariant_2026020200_HSURF.grib2.bz2
bzip2 -d icon_global_icosahedral_time-invariant_2026020200_HSURF.grib2.bz2
rm -rf icon_global_icosahedral_time-invariant_2026020200_HSURF.grib2.bz2
cdo -f nc copy icon_global_icosahedral_time-invariant_2026020200_HSURF.grib2 icon_global_icosahedral_time-invariant_2026020200_HSURF.nc
rm -rf icon_global_icosahedral_time-invariant_2026020200_HSURF.grib2

# horizontal grid:
wget http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_grid_0026_R03B07_G.nc

