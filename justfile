install *LIBS:
    pip install {{LIBS}} 
    pip freeze > requirements.txt