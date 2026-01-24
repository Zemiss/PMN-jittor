@echo off
echo 同步文件到 WSL...
echo.

wsl cp /mnt/c/Users/12445/Desktop/pybeta2.0/trainer_SID.py /home/x12345/py-wsl-beta/trainer_SID.py
wsl cp /mnt/c/Users/12445/Desktop/pybeta2.0/base_trainer.py /home/x12345/py-wsl-beta/base_trainer.py
wsl cp /mnt/c/Users/12445/Desktop/pybeta2.0/losses.py /home/x12345/py-wsl-beta/losses.py
wsl cp /mnt/c/Users/12445/Desktop/pybeta2.0/utils.py /home/x12345/py-wsl-beta/utils.py
wsl cp /mnt/c/Users/12445/Desktop/pybeta2.0/archs/ELD_models.py /home/x12345/py-wsl-beta/archs/ELD_models.py
wsl cp /mnt/c/Users/12445/Desktop/pybeta2.0/archs/modules.py /home/x12345/py-wsl-beta/archs/modules.py

echo.
echo 同步完成！
pause
