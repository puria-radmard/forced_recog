@echo off
REM Activate conda environment and test mock model

echo Activating conda environment...
call conda activate forced_recog

echo.
echo Running mock model test...
python scripts_Jesse/run_experiment.py --config configs/operationalizations/AT_2T/rec_config_mock.yaml

echo.
echo Mock model test complete!
pause

