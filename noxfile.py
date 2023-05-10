import nox
import sys


PYTHON = sys.executable

def compile_dependencies(session):
    session.run(PYTHON, "-m", "piptools", "compile", "requirements/dev.in")
    session.run(PYTHON, "-m", "piptools", "compile", "requirements/dev.in")

