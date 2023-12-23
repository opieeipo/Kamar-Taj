@echo off
REM python Windows setup
REM virtual setup
@mkdir .dev.windows.env
@mkdir .pilot.windows.env
REM build the environments
@py -m venv .dev.windows.env
@call .\.dev.windows.env\Scripts\activate.bat
@pip cache purge
@pip install -r requirements.txt
@call .\.dev.windows.env\Scripts\deactivate.bat
@py -m venv .pilot.windows.env
@call .\.pilot.windows.env\Scripts\activate.bat
@pip install -r requirements.txt
@call .\.pilot.windows.env\Scripts\deactivate.bat
@call .\.dev.windows.env\Scripts\activate.bat

