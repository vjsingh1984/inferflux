@echo off
setlocal

echo === Exporting ncu summary data ===

"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\ncu.bat" ^
  --import C:\Users\vjsin\code\inferflux\ncu_probe_admin.ncu-rep ^
  --csv ^
  --page details ^
  > C:\Users\vjsin\code\inferflux\ncu_details.csv 2>&1

echo === Export complete ===
