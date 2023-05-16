import nox
import sys


PYTHON = sys.executable

PACKAGE_NAME = "src/talking_python"


@nox.session
def compile_dependencies(session):
    session.run(PYTHON, "-m", "piptools", "compile", "requirements/dev.in")
    session.run(PYTHON, "-m", "piptools", "compile", "requirements/dev.in")


@nox.session
def format_package(session):
    session.run("black", PACKAGE_NAME)
    session.run("ruff", PACKAGE_NAME, "--fix")


@nox.session
def build_package(session):
    session.run(PYTHON, "-m", "build")


@nox.session
def install_package(session):
    session.run(PYTHON, "-m", "pip", "install", ".", "-w")
