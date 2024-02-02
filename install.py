from sys import executable
from subprocess import check_call

dependencies = [
    'numpy',
    'pandas',
    'scikit-learn',
    'matplotlib',
    'jupyter',
    'ghostml',
    'ctgan',
    'hmeasure',
    'imbalanced-learn',
    'scipy',
    'xgboost',
    'tabulate',
    'alive-progress',
    'gym',
    'keras-rl2',  # must appear before tensorflow dependencies
    'tensorflow-macos==2.12.0',
    'tensorflow-metal',
    'pydot',
    'simple_chalk',
]


if __name__ == "__main__":
    check_call([executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    for dependency in dependencies:
        check_call([executable, '-m', 'pip', 'install', dependency])
