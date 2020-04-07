from setuptools import setup

setup(name='CellNetwork',
      version='0.3',
      description='Library used for simulating cellular network diffusion',
      url='https://github.com/SirSharpest/cell_graph_diffusion',
      author='Nathan Hughes',
      author_email='nathan.hughes@jic.ac.uk',
      license='MIT',
      packages=['CellNetwork'],
      install_requires=['numpy',
                        'matplotlib',
                        'scipy',
                        'networkx'],
      zip_safe=True)
